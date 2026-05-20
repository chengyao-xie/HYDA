# main.py
from train import train_model, monitor_resources

if __name__ == "__main__":
    # Monitor system resources before starting training
    monitor_resources()

    # Start training
    train_model()

    # Monitor system resources again after training
    monitor_resources()