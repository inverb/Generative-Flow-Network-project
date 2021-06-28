# MNIST Loader Class
# Input: path t ofolder containing MNIST train and test files
# Output: Numpy arrays containing: train images, train labels, test images and test labels

class MnistLoader(object):
    def __init__(self, input_path):
        self.training_images_path = join(input_path, 'train-images.idx3-ubyte')
        self.training_labels_path = join(input_path, 'train-labels.idx1-ubyte')
        self.test_images_path     = join(input_path, 'test-images.idx3-ubyte')
        self.test_labels_path     = join(input_path, 'test-labels.idx1-ubyte')
    
    def read_file(self, images_path, labels_path):        
        images = []
        labels = []   
        
        with open(images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())
        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())           
        
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        im_train, la_train = self.read_file(self.training_images_path, self.training_labels_path)
        im_test, la_test = self.read_file(self.test_images_path, self.test_labels_path)
        return (im_train, la_train),(im_test, la_test)  
