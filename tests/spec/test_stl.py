import pytest

from d3rlpy.datasets import get_cartpole
from d3rlpy.spec.transform import CartpoleSTLTransform, CartpoleSTLAugment


@pytest.mark.parametrize("n_epochs", [1])
@pytest.mark.parametrize("experiment_name", ["cql_test"])
@pytest.mark.parametrize("test_size", [0.1])
def train_stl(n_epochs=10, experiment_name="cql", test_size=0.2):
    """Train on STL dataset using DiscreteCQL"""
    # get CartPole dataset
    dataset, env = get_cartpole()
    print("test_dataset", dataset)

    # Transform dataset
    transform = CartpoleSTLTransform()
    dataset_to_train = transform.get_shaped_dataset(dataset)


def test_augmented():
    """Test Augmented STL dataset"""
    # get CartPole dataset
    dataset, env = get_cartpole()
    print("test_dataset", dataset)

    transform = CartpoleSTLAugment()
    shaped_reward_episodes = transform.get_augmented_episodes(dataset, aug_options={'steps': 20, 'with_clamp': True})

    assert 1.1 * len(dataset.episodes) < len(shaped_reward_episodes) < 1.3 * len(dataset.episodes), \
        "Augmented dataset should have more episodes than original dataset and " \
        "less than 1.3 times more episodes by default"
