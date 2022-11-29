
# A Learned Pixel-by-Pixel Lossless Image Compression Method with 59K Parameters and Parallel Decoding




## Abstract
This paper considers lossless image compression and presents a learned compression system that can achieve state-of-the-art lossless compression performance but uses only 59K parameters, which is more than 30x less than other learned systems proposed recently in the literature. The explored system is based on a learned pixel-by-pixel lossless image compression method, where each pixel’s probability distribution parameters are obtained by processing the pixel’s causal neighborhood (i.e. previously encoded/decoded pixels) with a simple neural network comprising 59K parameters. This causality causes the decoder to operate sequentially, i.e. the neural network has to be evaluated for each pixel sequentially, which increases decoding time significantly with common GPU software and hardware. To reduce the decoding time, parallel decoding algorithms are proposed and implemented. The obtained lossless image compression system is compared to traditional and learned systems in the literature in terms of compression performance, encoding-decoding times and computational complexity. 
## Requirements
    pytorch
    PIL
    easydict
    numpy
    torch-vision
    torchnet
## Training
For training: 
1) Edit your json file.
2) In your json file, set "mode" as train.
3) Set "train_data" as the directory of training data.
## Real Compression
For real compression:
1) In your json file, set "mode" as test.
2) Set "run_mode" as encode.
3) Set "run_type" as "S" (sequential) or "P" for (parallel).
4) Set "decoder_method" as 1 (Wavefront parallel decoding method) or 2 (diagonal parallel decoding method).
5) Set "test_data" as the directory of test images.
## Decoding
For decoding operation:
1) In your json file, set "mode" as test.
2) Set "run_mode" as decode.
3) Set "run_type" as "S" (sequential) or "P" for (parallel). 
4) Set "bitstream_path" as the directory of decoded bitsream.
5) Set "H" and "W" as the height and width of the image to be decoded.
6) For parallel decoding, set "decoder_method" as 1 (Wavefront parallel decoding method) or 2 (diagonal parallel decoding method).
## How to run code
To run the code with adjusted settings:

    python main.py configs/your_json_file.json
## Arithmetic Coding
In our work, we used NumpyAc (Fast Autoregressive Arithmetic Coding) which is avaliable at github link:

https://github.com/zb12138/NumpyAc