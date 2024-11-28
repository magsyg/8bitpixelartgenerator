A simple pixel art generator which tries to convert an normal image to 8-bit
Will do the following thing to an image:
Downscale spatial dimension image to 64xN
Downscale available color to a range of 256 different values (8-bit)
Darken edges on the image

# How to run
Install requirements with:

```
pip install -r requirements.txt
```

Now run
```
python 8bitgenerator.py <path-to-folder-or-image>
```
Results will then be found in the results folder

# Results

![Pre-conversion](https://github.com/magsyg/8bitpixelartgenerator/blob/master/test/images/1.jpg =128x)
![After-converstion](https://github.com/magsyg/8bitpixelartgenerator/blob/master/test/results/pixel_1.png=128x)