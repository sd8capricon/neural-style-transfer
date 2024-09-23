# capture the high level detail of the content image i.e. reatin what the image is
content_layers = ["block5_conv2"]

# Capture details low,mid & high level details from the style image
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight = 1e-2
content_weight = 1e4
