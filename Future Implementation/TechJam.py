from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. BRANCH ADAPTER - Gets every branch to the same length AND the same normalized scale
#    Gets the input from all the branches ready for comparison
# =============================================================================

class BranchAdapter(nn.Module):
    def __init__(self, input_vector_size: int, shared_vector_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(input_vector_size, shared_vector_size)  
        self.normalization_layer = nn.LayerNorm(shared_vector_size)            

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        projected_vector = self.linear_layer(input_vector)
        normalized_vector = self.normalization_layer(projected_vector)
        return normalized_vector


# =============================================================================
# 2. FUSION HUB — concatenation + a single-hidden-layer MLP. 
# =============================================================================

@dataclass
class FusionHubConfig: #takes expected input and converges into per-branch shared dims, sized proportionally to each branch's information content instead of one uniform width
    clip_vector_size: int = 768
    dct_vector_size: int = 130
    efficientnet_vector_size: int = 1280

    clip_shared_size: int = 512
    dct_shared_size: int = 128
    efficientnet_shared_size: int = 768

    hidden_layer_size: int = 640

    number_of_branches: int = 3


class FusionHub(nn.Module): #Creation of three separate adapters and combining into one vector
    def __init__(self, config: FusionHubConfig):
        super().__init__()
        self.config = config

        self.clip_adapter = BranchAdapter(config.clip_vector_size, config.clip_shared_size)
        self.dct_adapter = BranchAdapter(config.dct_vector_size, config.dct_shared_size)
        self.efficientnet_adapter = BranchAdapter(config.efficientnet_vector_size, config.efficientnet_shared_size)

        combined_vector_size = config.clip_shared_size + config.dct_shared_size + config.efficientnet_shared_size  

        self.first_layer = nn.Linear(combined_vector_size, config.hidden_layer_size) # Produces 1 640-dim Vector
        self.output_layer = nn.Linear(config.hidden_layer_size, 1) #Produces 1 number (Rawe Score)

    def forward(
        self, #Inputting the 3 Branch Vectors + Reliability Tensor
        clip_vector: torch.Tensor,
        dct_vector: torch.Tensor,
        efficientnet_vector: torch.Tensor,
        branch_reliability_scores: Optional[torch.Tensor] = None,  
    ) -> dict:
        clip_features = self.clip_adapter(clip_vector)
        dct_features = self.dct_adapter(dct_vector)
        efficientnet_features = self.efficientnet_adapter(efficientnet_vector)

        if branch_reliability_scores is not None: #Dependent if there is input on each branch's reliability
            clip_features = clip_features * branch_reliability_scores[:, 0:1]
            dct_features = dct_features * branch_reliability_scores[:, 1:2]
            efficientnet_features = efficientnet_features * branch_reliability_scores[:, 2:3]

        combined_features = torch.cat([clip_features, dct_features, efficientnet_features], dim=-1) #Concatenation
        hidden_output = F.gelu(self.first_layer(combined_features)) # Detects contradiction between branches
        raw_score = self.output_layer(hidden_output).squeeze(-1)  #Uncalibrated Result

        return {"raw_score": raw_score}


# =============================================================================
# 3. TEMPERATURE SCALING 
# =============================================================================

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, raw_score: torch.Tensor) -> torch.Tensor: #Calibration
        return torch.sigmoid(raw_score / self.temperature) #Converts into 0-1 probability


def fit_temperature(raw_scores: torch.Tensor, true_labels: torch.Tensor,
                     max_iterations: int = 200, learning_rate: float = 0.01) -> TemperatureScaling:
    temperature_model = TemperatureScaling()
    optimizer = torch.optim.LBFGS([temperature_model.log_temperature],
                                   lr=learning_rate, max_iter=max_iterations)
    loss_function = nn.BCELoss() #Measuring how far the calibrated probabilities sit from the true 0/1 labels

    def compute_loss():
        optimizer.zero_grad()
        loss = loss_function(temperature_model(raw_scores), true_labels)
        loss.backward()
        return loss

    optimizer.step(compute_loss)
    return temperature_model


def expected_calibration_error(probabilities: torch.Tensor, true_labels: torch.Tensor, #comparison between claim and truth
                                number_of_bins: int = 10) -> float:
    probabilities = probabilities.detach().cpu()
    true_labels = true_labels.detach().cpu()

    bin_edges = torch.linspace(0, 1, number_of_bins + 1)
    total_calibration_error = 0.0
    total_images = len(probabilities)

    for i in range(number_of_bins):
        lower_bound, upper_bound = bin_edges[i], bin_edges[i + 1]
        images_in_this_bin = (
            (probabilities >= lower_bound)
            & (probabilities < upper_bound if i < number_of_bins - 1 else probabilities <= upper_bound)
        )
        images_in_bin_count = images_in_this_bin.sum().item()
        if images_in_bin_count == 0:
            continue
        average_confidence = probabilities[images_in_this_bin].mean().item()
        true_positive_rate = true_labels[images_in_this_bin].mean().item()
        total_calibration_error += (images_in_bin_count / total_images) * abs(average_confidence - true_positive_rate)

    return total_calibration_error


# =============================================================================
# 4. CACHING PIPELINE + REALITY CHECKER + FILE VALIDATION.
# =============================================================================

CLIP_COLUMN, DCT_COLUMN, EFFICIENTNET_COLUMN = 0, 1, 2 #Labelling the Branches


def filter_valid_images(image_paths: list, image_labels: list) -> tuple:
    
    from PIL import Image

    valid_image_paths, valid_image_labels = [], []
    skipped_images = []

    for image_path, image_label in zip(image_paths, image_labels): #Catching File Errors
        try:
            with Image.open(image_path) as opened_image:
                opened_image.verify()
            valid_image_paths.append(image_path)
            valid_image_labels.append(image_label)
        except Exception as error:
            skipped_images.append((image_path, str(error)))

    if skipped_images:
        print(f"WARNING: skipped {len(skipped_images)} unreadable image(s):")
        for image_path, reason in skipped_images[:10]:
            print(f"  {image_path}: {reason}")
        if len(skipped_images) > 10:
            print(f"  ... and {len(skipped_images) - 10} more")

    return valid_image_paths, valid_image_labels


def check_branch_reliability(image_paths: list, minimum_resolution: int = 256) -> torch.Tensor: #Catching shoftfalls in branches for different image types
  
    from PIL import Image

    reliability_scores = torch.ones(len(image_paths), 3)  # Starting with 1.0 for all Branches.

    for image_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as opened_image:
            image_format = (opened_image.format or "").upper()
            color_mode = opened_image.mode
            image_width, image_height = opened_image.size

        if image_format not in ("JPEG", "JPG"):
            reliability_scores[image_index, DCT_COLUMN] = 0.0        
        elif color_mode not in ("RGB", "YCbCr"):
            reliability_scores[image_index, DCT_COLUMN] *= 0.15       

        if min(image_width, image_height) < minimum_resolution:
            reliability_scores[image_index, EFFICIENTNET_COLUMN] *= 0.5

    return reliability_scores


def precompute_and_cache( #Caching is to store trained memory so the encoders do not have to run multiple times.
    image_paths: list,
    clip_encoder, dct_encoder, efficientnet_encoder,
    image_labels: list,
    cache_file_path: str = "cached_branch_vectors.pt",
    batch_size: int = 32,
    minimum_resolution: int = 256,
):
    image_paths, image_labels = filter_valid_images(image_paths, image_labels)

    clip_vectors, dct_vectors, efficientnet_vectors = [], [], []

    with torch.no_grad(): #Tells the system which images to skip
        for batch_start_index in range(0, len(image_paths), batch_size):
            image_paths_in_this_batch = image_paths[batch_start_index:batch_start_index + batch_size]

            clip_vectors.append(clip_encoder(image_paths_in_this_batch).cpu())
            dct_vectors.append(dct_encoder(image_paths_in_this_batch).cpu())
            efficientnet_vectors.append(efficientnet_encoder(image_paths_in_this_batch).cpu())

    reliability_scores = check_branch_reliability(image_paths, minimum_resolution=minimum_resolution)

    cached_data = {
        "clip_vectors": torch.cat(clip_vectors, dim=0),
        "dct_vectors": torch.cat(dct_vectors, dim=0),
        "efficientnet_vectors": torch.cat(efficientnet_vectors, dim=0),
        "reliability_scores": reliability_scores,
        "image_labels": torch.tensor(image_labels, dtype=torch.float32),
    }
    torch.save(cached_data, cache_file_path)
    print(f"Cached {len(image_paths)} examples to {cache_file_path}")


class CachedBranchDataset(Dataset):
    def __init__(self, cache_file_path: str):
        cached_data = torch.load(cache_file_path)
        self.clip_vectors = cached_data["clip_vectors"]
        self.dct_vectors = cached_data["dct_vectors"]
        self.efficientnet_vectors = cached_data["efficientnet_vectors"]
        self.image_labels = cached_data["image_labels"]
        # .get(...) with an all-ones fallback: gracefully handles a cache
        # file saved before the reality checker existed.
        self.reliability_scores = cached_data.get(
            "reliability_scores", torch.ones(len(self.image_labels), 3)
        )

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        return (
            self.clip_vectors[index], self.dct_vectors[index], self.efficientnet_vectors[index],
            self.reliability_scores[index], self.image_labels[index],
        )


# =============================================================================
# 5. TRAINING UTILITIES.
# =============================================================================

class EarlyStopping: #Stops the model from memorizing to keep it at minimal training loss
    def __init__(self, patience_in_epochs: int = 10, minimum_improvement: float = 1e-4):
        self.patience_in_epochs = patience_in_epochs
        self.minimum_improvement = minimum_improvement
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_weights = None

    def step(self, current_validation_loss: float, model: nn.Module) -> bool:
        if current_validation_loss < self.best_validation_loss - self.minimum_improvement:
            self.best_validation_loss = current_validation_loss
            self.epochs_without_improvement = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience_in_epochs

    def restore_best(self, model: nn.Module):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)


def smoothed_bce_loss(raw_scores: torch.Tensor, true_labels: torch.Tensor, #Pushing ther scores inward so it doesnt tend towards inifnity
                       smoothing_amount: float = 0.05,
                       positive_class_weight: Optional[torch.Tensor] = None):
    smoothed_labels = true_labels * (1 - smoothing_amount) + 0.5 * smoothing_amount
    return F.binary_cross_entropy_with_logits(raw_scores, smoothed_labels, pos_weight=positive_class_weight)


def compute_positive_class_weight(training_data_loader: DataLoader) -> torch.Tensor: #Makes the rare class count for more so the model treats every equally

    number_of_positive_examples, number_of_negative_examples = 0, 0
    for *_, labels_batch in training_data_loader:
        number_of_positive_examples += int((labels_batch == 1).sum().item())
        number_of_negative_examples += int((labels_batch == 0).sum().item())

    if number_of_positive_examples == 0:
        print("WARNING: no positive (AI) examples found in training data — "
              "positive_class_weight defaulting to 1.0 (no correction applied)")
        return torch.tensor(1.0)

    return torch.tensor(number_of_negative_examples / number_of_positive_examples)


def get_optimizer_parameter_groups(model: nn.Module, weight_decay_amount: float) -> list:
 
    parameters_with_decay, parameters_without_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1:  # every bias, and every LayerNorm weight+bias
            parameters_without_decay.append(parameter)
        else:
            parameters_with_decay.append(parameter)

    return [
        {"params": parameters_with_decay, "weight_decay": weight_decay_amount},
        {"params": parameters_without_decay, "weight_decay": 0.0},
    ]


def train_fusion_hub(
    model: FusionHub,
    training_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    number_of_epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay_amount: float = 0.04,
    label_smoothing_amount: float = 0.05,
    patience_in_epochs: int = 10,
    device: str = "cuda",
):
    model.to(device)

    positive_class_weight = compute_positive_class_weight(training_data_loader).to(device)
    print(f"computed positive_class_weight from training data: {positive_class_weight.item():.3f}")

    optimizer = torch.optim.AdamW(
        get_optimizer_parameter_groups(model, weight_decay_amount), lr=learning_rate
    )
    early_stopper = EarlyStopping(patience_in_epochs=patience_in_epochs)

    training_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch_number in range(number_of_epochs):
        model.train()
        total_training_loss = 0.0
        for clip_batch, dct_batch, efficientnet_batch, reliability_batch, labels_batch in training_data_loader:
            clip_batch, dct_batch, efficientnet_batch, reliability_batch, labels_batch = (
                clip_batch.to(device), dct_batch.to(device), efficientnet_batch.to(device),
                reliability_batch.to(device), labels_batch.to(device),
            )
            optimizer.zero_grad()

            model_output = model(clip_batch, dct_batch, efficientnet_batch,
                                  branch_reliability_scores=reliability_batch)
            loss = smoothed_bce_loss(model_output["raw_score"], labels_batch,
                                      label_smoothing_amount, positive_class_weight=positive_class_weight)

            loss.backward()
            optimizer.step()

            total_training_loss += loss.item() * labels_batch.size(0)

        average_training_loss = total_training_loss / len(training_data_loader.dataset)

        model.eval() #Actual Training Now
        total_validation_loss, number_correct = 0.0, 0
        with torch.no_grad():
            for clip_batch, dct_batch, efficientnet_batch, reliability_batch, labels_batch in validation_data_loader:
                clip_batch, dct_batch, efficientnet_batch, reliability_batch, labels_batch = (
                    clip_batch.to(device), dct_batch.to(device), efficientnet_batch.to(device),
                    reliability_batch.to(device), labels_batch.to(device),
                )
                model_output = model(clip_batch, dct_batch, efficientnet_batch,
                                      branch_reliability_scores=reliability_batch)
                loss = smoothed_bce_loss(model_output["raw_score"], labels_batch,
                                          label_smoothing_amount, positive_class_weight=positive_class_weight)
                total_validation_loss += loss.item() * labels_batch.size(0)
                predicted_labels = (torch.sigmoid(model_output["raw_score"]) > 0.5).float()
                number_correct += (predicted_labels == labels_batch).sum().item()

        average_validation_loss = total_validation_loss / len(validation_data_loader.dataset)
        validation_accuracy = number_correct / len(validation_data_loader.dataset)

        training_history["train_loss"].append(average_training_loss)
        training_history["val_loss"].append(average_validation_loss)
        training_history["val_accuracy"].append(validation_accuracy)

        print(f"epoch {epoch_number+1:3d} | train_loss {average_training_loss:.4f} | "
              f"val_loss {average_validation_loss:.4f} | val_accuracy {validation_accuracy:.4f}")

        if early_stopper.step(average_validation_loss, model):
            print(f"Early stopping at epoch {epoch_number+1} "
                  f"(best val_loss={early_stopper.best_validation_loss:.4f})")
            break

    early_stopper.restore_best(model)
    return model, training_history


# =============================================================================
# 6. CONFIDENCE BANDS + CALIBRATED FUSION HUB.
# =============================================================================

CONFIDENCE_BANDS = ( #Grouping them together into bands
    (0.00, 0.30, "Low Chance of AI", "Deemed as non-AI"),
    (0.30, 0.40, "Little-Moderate Chance of AI", "Deemed as non-AI"),
    (0.40, 0.70, "Moderate-High possibility of AI", "Deemed as AI"),
    (0.70, 1.01, "High Possibility of AI", "Deemed as AI"),
)


def classify_confidence_band(probability: float) -> dict:
    for lower_bound, upper_bound, band_name, recommended_action in CONFIDENCE_BANDS:
        if lower_bound <= probability < upper_bound:
            display_upper = int(upper_bound * 100) if upper_bound <= 1 else 100
            return {
                "band": band_name,
                "action": recommended_action,
                "range": f"{int(lower_bound*100)}-{display_upper}%",
            }
    raise ValueError(f"probability {probability} outside expected [0, 1] range")


class CalibratedFusionHub(nn.Module): #Combining both Fusion Hub and Temp Scaling
    def __init__(self, fusion_hub: FusionHub, temperature_calibrator: TemperatureScaling):
        super().__init__()
        self.fusion_hub = fusion_hub
        self.temperature_calibrator = temperature_calibrator

    @torch.no_grad()
    def forward(
        self,
        clip_vector: torch.Tensor,
        dct_vector: torch.Tensor,
        efficientnet_vector: torch.Tensor,
        branch_reliability_scores: Optional[torch.Tensor] = None,
        return_details: bool = True,
    ) -> dict:
        self.eval()

        fusion_output = self.fusion_hub(
            clip_vector, dct_vector, efficientnet_vector,
            branch_reliability_scores=branch_reliability_scores,
        )
        calibrated_probability = self.temperature_calibrator(fusion_output["raw_score"])

        result = {
            "probability": calibrated_probability,
            "raw_score": fusion_output["raw_score"],
            "temperature": self.temperature_calibrator.temperature.item(),
        }

        if return_details:
            band_info_per_image = [classify_confidence_band(p.item()) for p in calibrated_probability]
            result["band"] = [info["band"] for info in band_info_per_image]
            result["action"] = [info["action"] for info in band_info_per_image]

        return result

    def save_pretrained(self, save_path: str):
        torch.save({
            "fusion_hub_state": self.fusion_hub.state_dict(),
            "fusion_hub_config": self.fusion_hub.config,
            "temperature_calibrator_state": self.temperature_calibrator.state_dict(),
        }, save_path)

    @classmethod
    def load_pretrained(cls, save_path: str, device: str = "cpu") -> "CalibratedFusionHub":
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        fusion_hub = FusionHub(checkpoint["fusion_hub_config"])
        fusion_hub.load_state_dict(checkpoint["fusion_hub_state"])
        temperature_calibrator = TemperatureScaling()
        temperature_calibrator.load_state_dict(checkpoint["temperature_calibrator_state"])
        combined_model = cls(fusion_hub, temperature_calibrator)
        combined_model.to(device)
        combined_model.eval()
        return combined_model


def summarize_predictions( #Final Summary of all the Results
    calibrated_model: CalibratedFusionHub,
    data_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    from collections import Counter

    all_probabilities, all_band_labels = [], []

    for clip_batch, dct_batch, efficientnet_batch, reliability_batch, _labels_batch in data_loader:
        clip_batch, dct_batch, efficientnet_batch, reliability_batch = (
            clip_batch.to(device), dct_batch.to(device),
            efficientnet_batch.to(device), reliability_batch.to(device),
        )
        result = calibrated_model(clip_batch, dct_batch, efficientnet_batch,
                                   branch_reliability_scores=reliability_batch, return_details=True)
        all_probabilities.append(result["probability"].cpu())
        all_band_labels.extend(result["band"])

    all_probabilities = torch.cat(all_probabilities, dim=0)
    total_images = len(all_probabilities)
    band_counts = Counter(all_band_labels)

    band_display_order = ["Low Chance of AI", "Little-Moderate Chance of AI",
                           "Moderate-High possibility of AI", "High Possibility of AI"]
    column_width = 33

    print(f"=== Prediction summary across {total_images} images ===")
    print(f"{'band':{column_width}s} {'count':>8s} {'%':>7s}")
    for band_name in band_display_order:
        count = band_counts.get(band_name, 0)
        print(f"{band_name:{column_width}s} {count:8d} {100*count/total_images:6.1f}%")

    print(f"\n=== Individual image scores ===")
    for image_index, (probability, band_name) in enumerate(zip(all_probabilities, all_band_labels)):
        print(f"image {image_index:5d}: probability={probability.item():.3f} | band={band_name}")

    return {"probabilities": all_probabilities, "bands": all_band_labels, "counts": dict(band_counts)}