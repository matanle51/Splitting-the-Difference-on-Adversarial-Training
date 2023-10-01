# Splitting the Difference on Adversarial Training
This repository contains the code for the paper: "Splitting the Difference on Adversarial Training" which was accepted to USENIX Security 24'.

In the repository, you can find the training and evaluation code of the method presented in the paper, dubbed: Double Boundary Adversarial Training (DBAT).


## Prerequisites (preferable):

- python = 3.9.2
- torch = 2.0.1

## Code usages:

Running DBAT Training
```
python train_dbat.py
```

Running Auto-Attack [1]:
```
import test_auto_attack
natural_acc, robust_acc = test_auto_attack.main()
```


Running White-box attacks:
```
from whitebox_attack import eval_adv_test_whitebox
natural_acc, robust_acc = eval_adv_test_whitebox(model, device, test_loader, num_test_samples, 
                                                 epsilon, step_size, num_attack_steps, num_classes)
```

Running Black-box attacks:
```
from blackbox_attack import eval_adv_test_blackbox
natural_acc, robust_acc = eval_adv_test_blackbox(model_target, model_source, device, test_loader, num_test_samples,
                                                 epsilon, step_size, num_attack_steps, num_classes)
```


## References:

[1] Auto-Attack: https://github.com/fra31/auto-attack
