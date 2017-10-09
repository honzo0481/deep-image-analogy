import click
import cv2 as cv

from patchmatch import Patchmatch


@click.group()
@click.argument('style_image', type=click.Path(exists=True))
@click.argument('content_image', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, style_image, content_image):
    ctx.obj['style_image'] = style_image
    ctx.obj['content_image'] = content_image


@cli.command()
@click.option('--alpha', default=0.5, type=float,
              help="")
@click.option('--patchsize', default=3, type=int,
              help="Odd only")
@click.option('--iterations', default=5, type=int,
              help="Number of propogation and random search to run")
@click.option('--w', default=None, type=int,
              help="Maximum search radius")
@click.option('--save_images', is_flag=True,
              help="Save reconstructed images of source after each iteration")
@click.option('--save_dir', default="results",
              help="Directory to write images to")
@click.option('--v', is_flag=True,
              help="Print progress during propogation and random search")
@click.pass_context
def patchmatch(ctx, alpha, patchsize, iterations,
               w, save_images, save_dir, v):
    A = ctx.obj['style_image']
    B = ctx.obj['content_image']

    click.echo("Running Patchmatch...\n")
    click.echo("A: {}\tB: {}".format(A, B))

    A = cv.imread(A)
    B = cv.imread(B)

    pm = Patchmatch(A, B,
                    patchsize=patchsize,
                    alpha=alpha,
                    iterations=iterations,
                    w=w)

    pm.propagate_and_random_search(
        write_images=save_images,
        img_directory=save_dir,
        verbose=v)

if __name__ == "__main__":
    cli(obj={})
