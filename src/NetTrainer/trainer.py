class Trainer:
    """
    Trainer class for PyTorch models with phased/shard-based training.

    Args:
        model (nn.Module): PyTorch model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function.
        device (torch.device, optional): Device to run the model on.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: callable, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.losses: List[float] = []

    # ---------------- Sharding ---------------- #

    def shard_data(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split a tensor into shards based on a prime number near sqrt of dataset size.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of shards.
        """
        data = data.to(self.device)
        size = int(math.sqrt(len(data)))
        num_shards = self._nearest_prime(size)
        shard_size = len(data) // num_shards
        shards = [data[i*shard_size : (i+1)*shard_size] for i in range(num_shards - 1)]
        shards.append(data[(num_shards-1)*shard_size:])
        return shards

   # ---------------- Helper Functions  ---------------- #

    def _is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0: return False
        return True

    def _nearest_prime(self, n: int) -> int:
        offset = 0
        while True:
            if self._is_prime(n - offset): return n - offset
            if self._is_prime(n + offset): return n + offset
            offset += 1

    # ---------------- Training ---------------- #
    def train(self, x: torch.Tensor, y: torch.Tensor,
              epochs: int = 1000, reps: int = 1,
              phased: bool = False, plot: bool = True,
              reportProgress = True, limitToMaxAccuracy = False):
        """
        Train the model.

        Args:
            x (torch.Tensor): Input features.
            y (torch.Tensor): Target tensor.
            epochs (int): Number of epochs per repetition.
            reps (int): Number of repetitions.
            phased (bool): Whether to use shard-based phased training.
            plot (bool): Whether to plot loss curve.
        """
        x, y = x.to(self.device), y.to(self.device)

        try:
            self.model.load_state_dict(torch.load("model.pth"))
            print("Model loaded from checkpoint.")
        except FileNotFoundError:
            print("No checkpoint found, starting fresh.")

        if phased:
            x_shards = self.shard_data(x)
            y_shards = self.shard_data(y)
        else:
            x_shards, y_shards = [x], [y]

        self.model.train()
        try:
          for rep in range(reps):
              print(f"Repetition {rep+1}/{reps}")
              for shard_idx, (x_shard, y_shard) in enumerate(zip(x_shards, y_shards)):
                  for epoch in range(epochs):
                      self.optimizer.zero_grad()
                      pred = self.model(x_shard)
                      loss = self.loss_fn(pred, y_shard)
                      loss.backward()
                      self.optimizer.step()
                      self.losses.append(loss.item())

                      if epoch % max(1, epochs // 10) == 0 and reportProgress:
                          with torch.no_grad():
                              correct = torch.sum(torch.isclose(pred, y_shard, atol=0.1))

                              accuracy = (correct.item() / len(y_shard)) * 100

                              if accuracy == 100.00 and limitToMaxAccuracy:

                                break
                          print(f"Shard {shard_idx+1}/{len(x_shards)} - Epoch {epoch}/{epochs} "
                                f"- Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%")

                          print(f"Checkpoint saved after shard {shard_idx+1}")

                  torch.save(self.model.state_dict(), "model.pth")
        except KeyboardInterrupt as e:
          torch.save(self.model.state_dict(), "model.pth")
          print("Training interrupted!")

        except Exception as e:
          torch.save(self.model.state_dict(), "model.pth")
          print("Uh oh! something wrong occured, saved model!")
          print(e)


        finally:
          if plot:
              plt.figure(figsize=(8,5))
              plt.plot(self.losses)
              plt.xlabel("Iterations")
              plt.ylabel("Loss")
              plt.yscale("log")
              plt.grid(True)
              plt.show()

    # ---------------- Evaluation ---------------- #
    def evaluate(self, x: torch.Tensor):
        """
        Evaluate model predictions.

        Args:
            x (torch.Tensor): Input tensor.
        """
        self.model.eval()
        x = x.to(self.device)
        try:
            self.model.load_state_dict(torch.load("model.pth"))
        except FileNotFoundError:
            print("No checkpoint found for evaluation.")
            return

        with torch.no_grad():
            pred = self.model(x)
            for val in pred:
                print(val.item())

__all__ = ["Trainer"]
