from PIL import Image, ImageColor
import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
import skimage.transform as tf


class ImageMap:
    def __init__(self, in_map, mapping=None):
        if not isinstance(in_map, np.ndarray):
            in_map = plt.imread(in_map)[:, :, :3]  # RGBA -> RGB
        self.map = in_map
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]
        self.mapping = mapping

    def read_rgb(self):
        return Image.fromarray((self.map * 255).astype('uint8'), 'RGB')

    def read_l(self):
        return Image.fromarray((self.map * 255).astype('uint8'), 'L')

    def apply_mask(self, mask, weight):
        if len(self.map.shape) == 3:
            mask = mask[:, :, None]
        result = ImageMap(self.map * (1 - weight) + mask * weight)
        result.normalize()
        return result

    def apply_circular_mask(self, weight, n=1.25):
        interpolation = lambda x: x ** n
        mask = np.outer(
            self.create_gradient(self.height, f=interpolation, two_dir=True),
            self.create_gradient(self.width, f=interpolation, two_dir=True),
        )

        return self.apply_mask(mask, weight)

    def apply_square_mask(self, weight, edge_size):
        mask = np.ones([self.height, self.width])
        gradient = self.create_gradient(edge_size)

        for i in range(self.height):
            mask[i, :edge_size] *= gradient
            mask[i, self.width - edge_size :] *= gradient[::-1]
        for i in range(self.width):
            mask[:edge_size, i] *= gradient
            mask[self.height - edge_size :, i] *= gradient[::-1]

        return self.apply_mask(mask, weight)

    def create_gradient(self, size, f=lambda x: x, two_dir=False):
        """
        f : [0, 1] -> [0, 1]
        """
        gradient = np.zeros([size])
        if two_dir:
            size = size // 2
        for i in range(size):
            gradient[i] = f(i / size)
            if two_dir:
                gradient[-i - 1] = f(i / size)
        return gradient

    def resize(self, new_dims):
        return ImageMap(tf.resize(self.map, new_dims), self.mapping)

    def normalize(self):
        self.map -= np.min(self.map)
        self.map /= np.max(self.map)

    def colorize(self):
        colorized = Image.new('RGB', (self.width, self.height))
        for i in range(self.height):
            for j in range(self.width):
                for m in self.mapping:
                    if self.map[i, j] <= m.upper_bound:
                        colorized.putpixel((j, i), m.color)
                        break
        return colorized

    def texturize(self, blend_factor=0.08):
        texturized = np.zeros([self.height, self.width, 3])
        divisor = np.zeros_like(texturized)
        no_blend_count = sum([int(not m.blend) for m in self.mapping])

        for i, m in enumerate(self.mapping):
            mask = (m.upper_bound >= self.map).astype(int)
            mask *= (self.map >= m.lower_bound).astype(int)
            if i >= no_blend_count and blend_factor != 0:

                # generate boolean mask of edges

                # special cases: firsst and last layer
                if i < len(self.mapping):
                    blend_mask = (m.upper_bound + blend_factor >= self.map).astype(int)
                else:
                    blend_mask = np.ones_like(mask)
                if i >= no_blend_count + 1:
                    blend_mask *= (self.map >= (m.lower_bound - blend_factor)).astype(int)
                else:
                    blend_mask *= (self.map >= m.lower_bound).astype(int)

                blend_mask -= mask

                # make mask relative to edges
                blend_mask = blend_mask.astype(float)
                blend_mask *= self.map
                blend_mask[blend_mask != 0] -= (m.lower_bound + m.upper_bound) / 2
                blend_mask = abs(blend_mask)

                # normalize mask and transform 0s to 1s and 1s to 0s
                blend_mask[blend_mask != 0] -= np.min(blend_mask[blend_mask != 0])
                blend_mask /= np.max(blend_mask)
                blend_mask[blend_mask != 0] -= 1
                blend_mask *= -1

                mask = mask.astype(float)
                mask += blend_mask

            layer = m.texture.make_composite((self.height, self.width)).map
            texturized += layer * mask[:, :, None]
            divisor += mask[:, :, None]

        result = ImageMap(texturized)
        result.map /= divisor
        return result.read_rgb()

    def blank_like(self):
        return ImageMap(np.ones([self.height, self.width]))


class Texture:
    def __init__(self, path, block_size, copy_overlap=1):
        self.name = path.split('/')[-1].replace('.png', '')
        self.path = path
        self.original = ImageMap(self.path)
        self.block_size = block_size
        self.blocks = self._get_blocks(copy_overlap)

    def make_composite(self, size, paste_overlap=2):
        return ImageMap(self._create(size, paste_overlap))

    def _get_blocks(self, overlap_factor):
        blocks = []
        block_inc = int(self.block_size / overlap_factor)
        for i in range(0, self.original.height - self.block_size, block_inc):
            for j in range(0, self.original.width - self.block_size, block_inc):
                blocks.append(
                    self.original.map[i : i + self.block_size, j : j + self.block_size].astype(
                        np.float64
                    )
                )
        return blocks

    def random_sample(self):
        return self.blocks[int(np.random.rand() * len(self.blocks))]

    def _create(self, img_size, overlap_factor):
        img_size = [x + 2 for x in img_size]
        block_overlap = int(self.block_size / overlap_factor)
        img = np.zeros((img_size[0], img_size[1], 3))
        window = np.outer(np.hanning(self.block_size), np.hanning(self.block_size))
        divisor = np.zeros_like(img) + 1e-10

        def set_pixels(coords, incs, end):
            adj_window = window[: end[0], : end[1], None]
            adj_block = block[: end[0], : end[1]]
            img[coords[0] : coords[0] + incs[0], coords[1] : coords[1] + incs[1]] += (
                adj_window * adj_block
            )
            divisor[coords[0] : coords[0] + incs[0], coords[1] : coords[1] + incs[1]] += adj_window

        for i in range(0, img_size[1], block_overlap):
            for j in range(0, img_size[0], block_overlap):
                block = self.blocks[int(np.random.rand() * len(self.blocks))]

                # if on the bottom or right edges of the image, block must be cropped
                if i > img_size[1] - self.block_size or j > img_size[0] - self.block_size:
                    gap = [min(img_size[1] - i, self.block_size), min(img_size[0] - j, self.block_size)]
                    set_pixels([i, j], gap, gap)

                else:
                    set_pixels([i, j], [self.block_size] * 2, [self.block_size] * 2)

        return (img / divisor)[1:-1, 1:-1]


class NoiseMap(ImageMap):
    """
    Useful resources
    https://www.youtube.com/watch?v=eaXk97ujbPQ
    https://medium.com/@travall/procedural-2d-island-generation-noise-functions-13976bddeaf9
    https://www.redblobgames.com/maps/terrain-from-noise/
    """

    def __init__(self, dimensions, flatness=1, octaves=None, show_components=False):
        self.width = dimensions[0]
        self.height = dimensions[1]

        if octaves is None:
            self.octaves = int(np.log2(self.width))
        else:
            self.octaves = octaves

        self.show_components = show_components
        if self.show_components:
            self.layers = [Image.new('L', (self.width, self.height)) for _ in range(self.octaves)]

        self.generate_noise_map(flatness)

    def generate_noise_map(self, flatness):
        self.map = np.zeros([self.height, self.width])
        divisor = 0

        for n in range(self.octaves):
            simplex = OpenSimplex(int(np.random.rand() * 1e5))
            frequency = 2 ** n / 1e2
            amplitude = 1 / frequency
            divisor += amplitude

            for i in range(self.height):
                for j in range(self.width):
                    rand = simplex.noise2d(x=frequency * i, y=frequency * j)
                    self.map[i, j] += ((rand + 1) / 2) * amplitude
                    if self.show_components:
                        self.layers[n].putpixel((j, i), int(255 * ((rand + 1) / 2)))

        if self.show_components:
            for x in self.layers:
                x.show()
            quit()

        self.map /= divisor
        self.map = self.map ** flatness
        self.normalize()


class Mapping:
    biomes = None

    def __init__(self, lower_bound, upper_bound, color, name, blend=True):
        if not Mapping.biomes:
            Mapping.biomes = self.create_biomes()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self.blend = blend
        self.color = color if type(color) == tuple else ImageColor.getrgb(color)
        for b in Mapping.biomes:
            if b.name == name:
                self.texture = b
                break

    def create_biomes(self):
        biomes = []
        for biome in ['desert', 'grass', 'snow', 'stone', 'coast']:
            biomes.append(Texture('images/samples/' + biome + '.png', 10, copy_overlap=1.5))
        for biome in ['hills', 'forest']:
            biomes.append(Texture('images/samples/' + biome + '.png', 15))
        biomes.append(Texture('images/samples/ocean.png', 50))
        return biomes


class GeneratedIsland:
    def __init__(self, size, flatness):
        self.size = size
        self.terrain = NoiseMap(size, flatness=flatness)
        self.moisture = NoiseMap(size)

    def create_mapping(self, mapping):
        self.terrain.mapping = []
        for i in range(len(mapping)):
            m = mapping[i]
            lower_bound = 0.0 if i == 0 else mapping[i - 1][0]
            blend = False if m[2] in ['coast', 'ocean'] else True
            self.terrain.mapping.append(Mapping(lower_bound, m[0], m[1], m[2], blend))

    # def resize(self, new_dims):
    #    result = self.terrain.resize(new_dims)
    #    mask = ImageMap(result.map > self.terrain.mapping[0].upper_bound).resize(new_dims)
    #    print(mask.map)
    #    return ImageMap(result.map * (mask.map >= 1.0), self.terrain.mapping)


class BigIsland(GeneratedIsland):
    def __init__(self, size, flatness=0.5):
        super().__init__(size, flatness)
        self.shape = NoiseMap(size)

        self.shape = self.shape.apply_circular_mask(0.75)
        self.shape.map = (self.shape.map > 0.3).astype(int)  # convert into boolean array

        self.terrain = self.terrain.apply_circular_mask(0.4)
        self.terrain = self.terrain.apply_mask(self.moisture.map, 0.3)
        self.terrain.map *= self.shape.map

        super().create_mapping(
            [
                [0.3, '#135AD4', 'ocean'],
                [0.4, '#F1DA7A', 'desert'],
                [0.5, '#CF8C36', 'hills'],
                [0.6, '#0ADD08', 'grass'],
                [0.8, '#228B22', 'forest'],
                [0.9, '#516572', 'stone'],
                [1.0, '#FFFFFF', 'snow'],
            ]
        )


class SmallIsland(GeneratedIsland):
    def __init__(self, size, flatness=0.7):
        super().__init__(size, flatness)
        self.terrain = self.terrain.apply_circular_mask(0.75)
        self.moisture = self.moisture.apply_circular_mask(0.4)
        self.terrain = self.terrain.apply_mask(self.moisture.map, 0.4)

        super().create_mapping(
            [
                [0.4, '#135AD4', 'ocean'],
                [0.5, '#7BC8F6', 'coast'],
                [0.6, '#F1DA7A', 'desert'],
                [0.8, '#0ADD08', 'grass'],
                [0.9, '#228B22', 'forest'],
                [1.0, '#516572', 'stone'],
            ]
        )


class Continent:
    def __init__(self, name, path, coordinates, size=None):
        self.name = name
        self.image = ImageMap(path)
        if size:
            self.image = self.image.resize(size)
        self.coordinates = coordinates


class World:
    def __init__(self, width):
        self.width = width
        self.height = int(self.width / 2)
        self.image = Image.new('RGB', (self.width, self.height))

    def small(self):
        return self.image.resize((800, int(800 * 2 / 3)))

    def smooth_paste(self, inimage, coordinates, edge_size=None):
        if edge_size is None:
            edge_size = inimage.width // 40
        mask = inimage.blank_like().apply_square_mask(1, edge_size=edge_size)
        self.image.paste(inimage.read_rgb(), coordinates, mask=mask.read_l())


def main():
    island = BigIsland((200, 200))
    island.terrain.colorize().show()
    scaled_island = island.terrain.resize((1000, 1000))
    scaled_island.texturize(0).show()
    scaled_island.texturize().show()

    # world = stitch_world_map()
    # world.image.show()


if __name__ == '__main__':
    main()
