# ReliableAI
터미널에 python test.py를 실행하면 자동으로 (MNIST/ CIFAR10), (target/untarget), (FGSM/PGD)경우에 대해 adversarial attack 결과를 출력한다.

test.py 파일의 eps, eps_step, k 값을 조정함으로써 하이퍼파라미터 세팅이 가능하다.

출력예시 : 
```text
==================================================
Processing dataset: cifar10
Attack method: targeted fgsm
Attack success rate: 12.50%
==================================================
Processing dataset: cifar10
Attack method: targeted pgd
Attack success rate: 100.00%
==================================================
Processing dataset: cifar10
Attack method: untargeted fgsm
Attack success rate: 89.06%
==================================================
Processing dataset: cifar10
Attack method: untargeted pgd
Attack success rate: 100.00%
==================================================
Processing dataset: mnist
Attack method: targeted fgsm
Attack success rate: 62.50%
==================================================
Processing dataset: mnist
Attack method: targeted pgd
Attack success rate: 90.62%
==================================================
Processing dataset: mnist
Attack method: untargeted fgsm
Attack success rate: 99.22%
==================================================
Processing dataset: mnist
Attack method: untargeted pgd
Attack success rate: 99.22%
```