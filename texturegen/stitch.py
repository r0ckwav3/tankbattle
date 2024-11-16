from PIL import Image
import os


def main():
    tilemap_width, tilemap_height = 1000, 1000
    texture_width, texture_height = None, None

    result = Image.new('RGBA', (tilemap_width, tilemap_height))
    textures = [t for t in os.listdir('./isometric') if t[-6:] != 'import']

    for i, texture in enumerate(textures):
        if texture[-6:] == 'import':
            continue

        im = Image.open(f'./isometric/{texture}')

        if not texture_width:
            texture_width, texture_height = im.size
            horizontal_textures = int(tilemap_width / texture_width)

        result.paste(im, (
            (i % horizontal_textures) * texture_width, int(i / horizontal_textures) * texture_height)
        )

    result.save('./atlas.png')


if __name__ == '__main__':
    main()
