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

<img src="https://github.com/magsyg/8bitpixelartgenerator/blob/master/test/images/2.jpg" width="200">
<img src="https://github.com/magsyg/8bitpixelartgenerator/blob/master/test/results/pixel_2.png" width="200">
