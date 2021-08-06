
import tf_model

class Resnet152x4(tf_model.ClassificationModel):

    def infer(self, image):
        image = self.transform(image)
        prediction = self.model(image).numpy()[0]
        prediction = tf.nn.sigmoid(prediction)

        return prediction

    def transform(self, image):
        image = np.array(image)
        image = tf.expand_dims(image, axis=0)
        return image
