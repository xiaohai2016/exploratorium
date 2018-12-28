"""
The Pytorch IWSLT operational main file
"""
from pytorch_iwslt_training import IWSLTTrainer

def train_on_cpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer(batch_size=6000)
    trainer.run_training(10, log_interval=1)

def train_on_one_gpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer(use_gpu=True, devices=[0], batch_size=6000)
    trainer.run_training(10, log_interval=1)

def train_on_two_gpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer(use_gpu=True, devices=[0, 1])
    trainer.run_training(10, log_interval=1)

def train_on_four_gpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer(use_gpu=True, devices=[0, 1, 2, 3])
    trainer.run_training(10, log_interval=1)

if __name__ == "__main__":
    train_on_cpu()
