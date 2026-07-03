
# New Ideas Inbox

This file is a place for Sidney to drop in new ideas for the project. There probably will end up falling into P4 workplan category.





# Research Idea: Scheduled Normalization for ImageNet-to-IFCB Domain Transfer

## Summary

This experiment tests whether an image normalization schedule can improve transfer learning from ImageNet-pretrained models to IFCB/NES plankton imagery.

Instead of choosing either fixed ImageNet normalization or fixed IFCB dataset normalization, the model would begin training with ImageNet normalization and gradually transition toward IFCB-specific normalization over the first epoch or first few warmup epochs.

The goal is to reduce the initial distribution shock experienced by an ImageNet-pretrained backbone while still allowing the model to adapt to the actual brightness, contrast, and grayscale-like statistics of the IFCB image domain.

## Motivation

ImageNet-pretrained backbones are usually trained with ImageNet normalization values:

```
mean: [0.485, 0.456, 0.406]
std:  [0.229, 0.224, 0.225]
```

The IFCB/NES dataset has substantially different estimated image statistics:

```
mean: [0.6425475478172302, 0.6425475478172302, 0.6425475478172302]
std:  [0.19147475063800812, 0.19147475063800812, 0.19147475063800812]
```

Qualitatively, this suggests the IFCB images are brighter, lower contrast, and effectively grayscale or near-grayscale compared with ImageNet natural images.

When using ImageNet-pretrained weights, there is a tension between two reasonable preprocessing choices.

ImageNet normalization preserves the input convention expected by the pretrained backbone. This may make the pretrained filters more immediately useful at the start of fine-tuning.

IFCB-specific normalization better centers and scales the actual target-domain data. This may improve adaptation to the microscopy/plankton domain, especially once the backbone is being fine-tuned end to end.

The proposed normalization scheduler is intended to combine both advantages.

## Core Hypothesis

A gradual transition from ImageNet normalization to IFCB normalization may improve transfer learning by giving the pretrained backbone an easier initial optimization problem while progressively adapting the model to the target image domain.

More formally:

A pretrained model may benefit from seeing ImageNet-normalized inputs early in training because its initial filters and activations were calibrated under that convention. As training progresses, gradually shifting toward IFCB-specific normalization may help the model learn a representation better matched to the actual brightness and contrast structure of IFCB images.

## Proposed Method

Let alpha be a schedule value between 0 and 1.

At the start of training:

```
alpha = 0
```

The model uses ImageNet normalization.

At the end of the warmup period:

```
alpha = 1
```

The model uses IFCB normalization.

The scheduled mean and standard deviation are computed by linear interpolation:

```
scheduled_mean = (1 - alpha) * imagenet_mean + alpha * ifcb_mean
scheduled_std  = (1 - alpha) * imagenet_std  + alpha * ifcb_std
```

Where:

```
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

ifcb_mean = [0.6425475478172302, 0.6425475478172302, 0.6425475478172302]
ifcb_std  = [0.19147475063800812, 0.19147475063800812, 0.19147475063800812]
```

The schedule can be applied per batch using global step:

```
alpha = min(1.0, global_step / warmup_steps)
```

Or per epoch:

```
alpha = min(1.0, current_epoch / warmup_epochs)
```

A batch-level schedule is smoother and may be preferable for initial testing.

## Interpretation

At alpha equals 0, the model receives images normalized exactly as an ImageNet-pretrained model expects.

At alpha equals 1, the model receives images normalized according to IFCB dataset statistics.

Intermediate alpha values gradually shift the input distribution from ImageNet-like preprocessing to target-domain preprocessing.

Because the IFCB mean is higher than the ImageNet mean, the schedule increasingly subtracts a brighter baseline from the image. Because the IFCB standard deviation is lower than ImageNet’s, the schedule increasingly amplifies deviations from that brighter background.

In practical terms, the scheduled normalization may gradually recenter the pale microscopy background while increasing the relative contrast of organism structures against that background.

## Why This Might Help

The method may help because pretrained models are sensitive to input centering and scaling. A sudden mismatch between pretraining normalization and fine-tuning normalization can shift early-layer activations away from the distribution the model initially expects.

A normalization schedule may reduce that shock.

Potential benefits include:

* Smoother early fine-tuning
* Less abrupt activation shift in early layers
* More stable loss during the first epoch
* Better use of ImageNet-pretrained features at step zero
* Gradual adaptation from natural-image statistics to microscopy-image statistics
* Reduced need for extremely conservative learning rates during early fine-tuning
* Possible improvement on rare or low-contrast plankton classes

This is especially plausible when the backbone is pretrained and fine-tuned end to end.

## Why This Might Not Help

The method also has risks.

Normalization is usually considered part of the data definition rather than a train-time augmentation. If the normalization changes during training, the model is optimizing against a moving input distribution.

Potential downsides include:

* The model may chase a shifting target distribution
* Early batches and later batches are not normalized consistently
* Batch normalization or related internal statistics may adapt unpredictably
* The schedule may slow convergence if the transition period is too long
* Fixed ImageNet normalization may already be sufficient for transfer learning
* Fixed IFCB normalization may already be sufficient if the backbone adapts quickly
* Added complexity may not translate into better final validation performance

The method should therefore be treated as an empirical ablation rather than an assumed improvement.

## When This Is Most Likely to Help

This idea is most likely to be useful under the following conditions:

* The model starts from ImageNet-pretrained weights
* The backbone is fine-tuned, not kept completely frozen
* The target image statistics differ meaningfully from ImageNet
* Early training is unstable or unusually slow
* The dataset is not large enough to make transfer learning irrelevant
* The learning rate is high enough that a hard input-distribution shift could matter
* The target domain is visually different but still shares useful low-level visual structure with ImageNet

IFCB plankton imagery appears to be a plausible candidate because it is brighter, lower contrast, and less chromatic than ImageNet imagery while still containing edges, contours, shapes, and textures that pretrained convolutional filters may usefully transfer.

## When This Is Less Likely to Help

The method is less likely to help under the following conditions:

* The model is trained from scratch
* The model uses no pretrained weights
* The ImageNet backbone is frozen and cannot adapt
* The training learning rate is already very low
* Training is already stable in the first few epochs
* The dataset is large enough that initialization matters less
* The backbone uses frozen normalization layers that expect ImageNet-like activations

If the backbone is frozen, shifting away from ImageNet normalization may be harmful. A frozen ImageNet-pretrained feature extractor generally cannot adapt its filters to the new input convention, so fixed ImageNet normalization is likely the safer default.

## Experimental Design

The proposed experiment should compare at least four conditions.

### A. Pretrained Backbone With Fixed ImageNet Normalization

Purpose:

This is the standard transfer learning baseline.

Expected behavior:

Strong early performance, stable use of pretrained filters, possibly less target-domain-specific input centering.

Configuration:

```
weights: ImageNet pretrained
normalization: ImageNet fixed
backbone: fine-tuned
```

### B. Pretrained Backbone With Fixed IFCB Normalization

Purpose:

Tests whether target-domain normalization is better than preserving the ImageNet input convention.

Expected behavior:

May adapt better to IFCB brightness and contrast, but may produce a larger activation shift at the start of fine-tuning.

Configuration:

```
weights: ImageNet pretrained
normalization: IFCB fixed
backbone: fine-tuned
```

### C. Pretrained Backbone With Scheduled ImageNet-to-IFCB Normalization

Purpose:

Tests the proposed curriculum-style transition.

Expected behavior:

May combine stable early transfer with better later target-domain centering.

Configuration:

```
weights: ImageNet pretrained
normalization: scheduled from ImageNet to IFCB
backbone: fine-tuned
schedule: first epoch or first few warmup epochs
```

### D. Random Initialization With Fixed IFCB Normalization

Purpose:

Measures the value of ImageNet pretraining itself.

Expected behavior:

May train more slowly, but provides an important control for whether ImageNet features are useful in this domain.

Configuration:

```
weights: none
normalization: IFCB fixed
backbone: trained from scratch
```

## Optional Additional Ablations

Additional useful experiments could include:

* Pretrained backbone with frozen feature extractor and ImageNet normalization
* Pretrained backbone with frozen feature extractor and IFCB normalization
* Pretrained backbone with scheduled normalization over 0.5 epoch
* Pretrained backbone with scheduled normalization over 1 epoch
* Pretrained backbone with scheduled normalization over 3 epochs
* Pretrained backbone with scheduled normalization over 5 epochs
* Nonlinear schedules such as cosine, sigmoid, or exponential interpolation
* Mean-only scheduling with fixed ImageNet standard deviation
* Standard-deviation-only scheduling with fixed ImageNet mean
* Per-channel IFCB statistics if future dataset estimates show non-identical channels

## Suggested Initial Schedule

A conservative first test would use a short linear warmup over the first epoch.

Rationale:

A one-epoch schedule is long enough to avoid a hard step-change in activation statistics, but short enough that most of training happens under the final target-domain normalization.

Suggested schedule:

```
warmup_epochs: 1
schedule_type: linear
start_normalization: ImageNet
end_normalization: IFCB
```

If the dataset is large and one epoch contains many optimization steps, a fractional-epoch schedule may also be worth testing.

Alternative schedule:

```
warmup_fraction: 0.25 to 0.5 epochs
```

If training is unstable or the first epoch loss curve is noisy, a longer schedule may be tested.

Alternative schedule:

```
warmup_epochs: 3
```

## Metrics to Watch

The main evaluation metric should remain validation macro-F1, especially if class imbalance is important.

Primary metrics:

* Validation macro-F1
* Validation micro-F1
* Validation accuracy
* Validation loss

Secondary diagnostics:

* Training loss smoothness during early batches
* Validation macro-F1 after epoch 1
* Time to reach a given validation macro-F1 threshold
* Per-class F1 for rare classes
* Per-class F1 for visually low-contrast classes
* Confusion matrix changes
* Stability across random seeds
* Gradient norm behavior during early fine-tuning
* Activation statistics in early layers, if easy to log

The scheduler may show its clearest benefit early in training rather than only in final performance. Therefore, early-epoch curves should be inspected carefully.

## Expected Outcomes

### Outcome 1: Scheduled Normalization Improves Early Training and Final F1

This would support the hypothesis that gradual input-distribution adaptation improves ImageNet-to-IFCB transfer.

Interpretation:

The model benefits from starting near the pretrained input convention and then adapting to target-domain statistics.

Next steps:

Test different schedule lengths, run multiple seeds, and inspect class-level gains.

### Outcome 2: Scheduled Normalization Improves Early Training but Not Final F1

This would suggest the schedule acts mainly as an optimization stabilizer.

Interpretation:

The model can eventually adapt under fixed normalization, but the schedule may make early training smoother.

Next steps:

Consider whether faster convergence or more stable training is valuable enough to justify the added complexity.

### Outcome 3: Fixed ImageNet Normalization Wins

This would suggest that preserving the pretrained model’s original input convention matters more than matching IFCB dataset statistics.

Interpretation:

The pretrained backbone may remain best calibrated under ImageNet normalization, even during fine-tuning.

Next steps:

Use ImageNet normalization as the default for pretrained runs.

### Outcome 4: Fixed IFCB Normalization Wins

This would suggest the target-domain statistics matter more than the original pretraining convention.

Interpretation:

The backbone adapts quickly enough that starting from the target-domain normalization is best.

Next steps:

Use IFCB normalization as the default for both scratch and pretrained runs.

### Outcome 5: Scratch Training Matches or Beats Pretraining

This would suggest ImageNet pretraining may not be very helpful for this dataset or model setup.

Interpretation:

The domain gap may be large enough, or the dataset large enough, that ImageNet initialization provides limited benefit.

Next steps:

Focus on domain-specific pretraining, self-supervised learning, or larger-scale IFCB/NES pretraining.

## Important Implementation Considerations

The scheduled normalization should be implemented as part of the transform pipeline or dataloader preprocessing, not as a random augmentation.

It should be deterministic with respect to global step or epoch.

The current alpha value should be logged during training so runs are reproducible and diagnosable.

The resolved config should record:

* Start mean
* Start standard deviation
* End mean
* End standard deviation
* Schedule type
* Warmup steps or epochs
* Whether interpolation is per step or per epoch

The training output should make it clear that normalization was scheduled, because this affects interpretation of the run.

## Possible Config Concept

A possible future config shape could look like this:

```
transforms:
  image_mode: rgb
  input_bit_depth: 8
  pipeline:
    - name: aspect_bucket
      buckets:
        - name: tall
          max_aspect: 0.8
          canvas_size: [320, 224]
        - name: square
          min_aspect: 0.8
          max_aspect: 1.25
          canvas_size: [224, 224]
        - name: wide
          min_aspect: 1.25
          max_aspect: 2.5
          canvas_size: [224, 320]
        - name: ultrawide
          min_aspect: 2.5
          canvas_size: [224, 448]
    - name: horizontal_flip
      p: 0.5
      train_only: true
    - name: vertical_flip
      p: 0.5
      train_only: true
    - name: scheduled_normalize
      schedule:
        type: linear
        warmup_epochs: 1
      start:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      end:
        mean: [0.6425475478172302, 0.6425475478172302, 0.6425475478172302]
        std: [0.19147475063800812, 0.19147475063800812, 0.19147475063800812]
```

For inference, the model should use the final normalization values, not the scheduled starting values.

Inference normalization:

```
mean: [0.6425475478172302, 0.6425475478172302, 0.6425475478172302]
std:  [0.19147475063800812, 0.19147475063800812, 0.19147475063800812]
```

This is important because the trained model ends training under the IFCB normalization convention.

## Open Questions

Several details should be tested rather than assumed.

Key questions:

* Should the schedule last for one epoch, several epochs, or only a fraction of an epoch?
* Should both mean and standard deviation be scheduled?
* Is mean scheduling more important than standard deviation scheduling?
* Does the method still help if batch normalization layers are frozen?
* Does the method help more with higher learning rates?
* Does the method help rare classes more than common classes?
* Does scheduled normalization improve final generalization or only early optimization?
* Is the effect consistent across random seeds?
* Does the best schedule depend on whether the backbone is frozen, partially unfrozen, or fully fine-tuned?

## Practical Recommendation

This idea is worth testing as an ablation, especially if using ImageNet-pretrained EfficientNet weights on IFCB/NES imagery.

The most useful initial experiment would be:

* Pretrained EfficientNet-B0 with fixed ImageNet normalization
* Pretrained EfficientNet-B0 with fixed IFCB normalization
* Pretrained EfficientNet-B0 with one-epoch linear ImageNet-to-IFCB normalization scheduling
* Randomly initialized EfficientNet-B0 with fixed IFCB normalization

The scheduled method should not replace the simpler baselines until it demonstrates clear value.

A successful result would be smoother early training, faster macro-F1 improvement, better final validation macro-F1, or improved rare-class performance compared with both fixed-normalization pretrained baselines.

## Short Name

Possible names for this method:

* Scheduled Normalization Transfer
* Normalization Warmup
* Input-Distribution Warmup
* Domain Normalization Scheduling
* ImageNet-to-IFCB Normalization Curriculum
* Normalization Curriculum for Domain Transfer

## One-Sentence Description

Scheduled Normalization Transfer gradually shifts image normalization from ImageNet statistics to IFCB dataset statistics during early fine-tuning, aiming to preserve pretrained feature usefulness at initialization while adapting the model to the target microscopy image domain.
