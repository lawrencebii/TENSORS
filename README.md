# TENSORS

# CONVOLUTIONS
this is a filter of weights that are used to multiply a pixel with its neighbours
to get a new value for the pixel
best filters to match inputs and outputs can be learnt over time
# POOLING
process of eliminating pixels in your image while maintaining senantics of the content
within the image.
max_pooling -> picking the most dense pixel among the 4 pixels

NB ..There are other approaches to pooling, such as min pooling, which
takes the smallest pixel value from the pool, and average pooling,
which takes the overall average value.
# BUILDING A CNN TO DISTINGUISH BETWEEN HORSES AND HUMAN
USING ImageDataGenerator -> ensure that the directory has 
the set of the named subdirectories
