from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras import metrics
from keras import losses
from keras import models, layers,Input
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import keras
import tensorflow as tf

# def preprocess_image(image):

#     # throw error if image is not valid
#     if image is None:
#         raise ValueError('Invalid image: Check the image path and try again.')
    
#     # load image and resize
#     img = load_img(image, target_size=(224, 224))
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#     return img_array

def load_classifier_model(model_path):
    model = models.load_model(model_path)
    return model

def predict(image, model):
    #image = preprocess_image(image)
    preds = model.predict(image)
    return preds
    
def model_setup():
    cut_off = 0.5
    metricsVec = [
        metrics.BinaryAccuracy(threshold=cut_off),
        metrics.Precision(thresholds=cut_off),
        metrics.Recall(thresholds=cut_off),
        metrics.AUC(),
    #     CohenKappa(num_classes=2),
    #     cohen_kappa_score,
    #     kappa_loss,
    #     cohen_kappa,
    #     cohen_kappa,
    #     tf.python.ops.metric_ops.cohen_kappa,
    ]
    model = load_classifier_model('densenetmodel.keras')
    Top_Layers = model.get_layer('Top_Layers')
    Output_layer = model.get_layer('Base_DenseNet201').get_layer('densenet201')
    Output_layer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001*3),
    loss=losses.BinaryCrossentropy(from_logits=False),
    metrics=metricsVec)
    return Top_Layers, Output_layer

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    cln =['global_average_pooling2d_2','dropout_2','dense_2']
    last_conv_layer_name = "conv5_block32_concat"
    classifier_layer_names = ["bn","relu"]
    conv_layer = model.get_layer(last_conv_layer_name)
    # conv_layer = model.get_layer('Base_DenseNet201').get_layer('densenet201').get_layer(last_conv_layer_name)
    grad_model = keras.models.Model(
        model.inputs, conv_layer.output)

    top_layer,_ = model_setup()
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    for layer_name in cln:
        x = top_layer.get_layer(layer_name)(x)    
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
     # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index.numpy()

def grad_cam(image, model, last_conv_layer_name, classifier_layer_names=None):

    _, model = model_setup()

    #image = preprocess_image(image)

    heatmap, top_pred_index = make_gradcam_heatmap(
        image, model, last_conv_layer_name, classifier_layer_names
    )
    
    image = tf.squeeze(image, axis=0)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    #jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


def main():
    temp ='temp'


if __name__ == '__main__':
    main()