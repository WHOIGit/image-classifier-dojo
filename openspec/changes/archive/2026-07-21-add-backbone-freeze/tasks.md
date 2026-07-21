## 1. Freeze policy

- [x] 1.1 Extend `BackboneFreezeConfig.policy` to `Literal["none", "frozen"]`
- [x] 1.2 In `apply_freeze_policy`, set `requires_grad=False` on backbone
      parameters when policy is `frozen`
- [x] 1.3 Filter frozen parameters out of the AdamW optimizer in
      `SupervisedTaskModule.configure_optimizers`
- [x] 1.4 Verification: compose `p2/07_transfer_miniset/frozen` and
      `p2/08_transfer_denticle_type/frozen_linear` via `dojo inspect config`
      (freeze policy resolves; frozen params excluded from the optimizer)
