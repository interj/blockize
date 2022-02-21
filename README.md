# K-means solution for translating pictures into JSON 3D rectangle collection

### [Original picture](https://commons.wikimedia.org/wiki/Landscape#/media/File:Che_ne_saj.jpg):
![Original](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Che_ne_saj.jpg/640px-Che_ne_saj.jpg?download)

### Downscaled, with k-means reduced color space (upper bound was 70'000 pixels):
![Flattened](https://i.imgur.com/WG1mV4R.jpg)

### Picture represented in 3D space using various sized rectangles (6116 rectangles total):
![3D](https://i.imgur.com/m7KQXye.png) ![Selected](https://i.imgur.com/YhRLuDw.png)

Caveats: alpha channel doesn't really work at the moment, compression parameters are a guesswork, could be improved using simmulated annealing