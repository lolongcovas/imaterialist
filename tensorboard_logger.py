from tensorboardX import SummaryWriter
import torchvision


def visualize_tensors(writer, tensors, tag, iteration):
    """
    Visualize tensors

    Args:
        writer (tensorboard SummaryWriter)
        tensors (torch tensor): NxCxHxW
    """

    image = torchvision.utils.make_grid(tensors, normalize=True, scale_each=True)

    writer.add_image(tag, image, iteration)


def visualize_histogram(writer, model, iteration):
    for name, param in model.named_parameters():
        if name.find('weight') >= 0:
            writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)
