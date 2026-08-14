import unittest

try:
    import torch
    import network

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - environment guard
    torch = None
    network = None
    HAS_TORCH = False


class _FakeReduceOp:
    SUM = object()


class _FakeDist:
    ReduceOp = _FakeReduceOp

    def __init__(self, peer_tensors):
        self.peer_tensors = list(peer_tensors)
        self.calls = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def all_reduce(self, tensor, op=None):
        tensor.add_(self.peer_tensors[self.calls].to(tensor))
        self.calls += 1


@unittest.skipUnless(HAS_TORCH, "torch is required for DPGA runtime tests")
class DPGAGlobalDDPTest(unittest.TestCase):
    def test_allreduce_mean_averages_raw_gradient_list(self):
        original_dist = network.dist
        fake_dist = _FakeDist(
            [
                torch.tensor([5.0, 7.0, 4.0]),
            ]
        )
        try:
            network.dist = fake_dist
            averaged = network._dpga_allreduce_mean(
                [
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([2.0]),
                ]
            )
        finally:
            network.dist = original_dist

        self.assertEqual(fake_dist.calls, 1)
        self.assertTrue(torch.allclose(averaged[0], torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.allclose(averaged[1], torch.tensor([3.0])))

    def test_allreduce_mean_splits_bounded_buckets(self):
        original_dist = network.dist
        fake_dist = _FakeDist(
            [
                torch.tensor([3.0, 5.0]),
                torch.tensor([7.0]),
            ]
        )
        try:
            network.dist = fake_dist
            averaged = network._dpga_allreduce_mean(
                [
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([5.0]),
                ],
                max_bucket_bytes=8,
            )
        finally:
            network.dist = original_dist

        self.assertEqual(fake_dist.calls, 2)
        self.assertTrue(torch.allclose(averaged[0], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(averaged[1], torch.tensor([6.0])))


if __name__ == "__main__":
    unittest.main()
