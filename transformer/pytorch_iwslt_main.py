"""
The Pytorch IWSLT operational main file
"""
from pytorch_iwslt_training import IWSLTTrainer

def train_on_cpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer()
    trainer.run_training(10, log_interval=1)

def train_on_one_gpu():
    """Run training for IWSLT on CPU for debugging purpose"""
    trainer = IWSLTTrainer(use_gpu=True, devices=[0])
    trainer.run_training(10, log_interval=1)

if __name__ == "__main__":
    train_on_cpu()