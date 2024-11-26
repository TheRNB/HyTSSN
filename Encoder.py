import pymonntorch as pymo
from pymonntorch import np
from PIL import Image
import matplotlib.pyplot as plt

def timeToFirstSpike(image, timeframe=20):
    #TODO: migrate to PyTorch
    """encodes grayscale image into spike train using TTFS
        NOTE: images are in grayscale.
    Args:
        image (np.ndarray): the input image.
        timeframe (int, optional): The number of timeframe the image values have,
                                Defaults to 20 with a unified distribution.

    Returns:
        numpy.ndarray: Time-to-first-spike encoded image.
    """

    encoded_image = np.zeros((image.shape[0], timeframe))
    #time_resolution = 255.00 / timeframe
    for i in range(image.shape[0]):
        ttfs = int(timeframe - np.floor((image[i] * timeframe) / 255.00))-1
        encoded_image[i,ttfs] = 1

    return encoded_image

def positionalEncoding(image, steps=20, std=1, timeframe=20):
    #TODO: migrate to PyTorch
    """encodes grayscale image into spike train using positional encoding with standard distribution.
        NOTE: images are in grayscale.

    Args:
        image (np.ndarray): the input image.
        steps (int, optional): the number of distriubutions.
        std (int, optional): the standard deviation of the normal distributions.
        timeframe (int, optional): the time span that the encoding should be fed to the network.

    Returns:
        np.ndarray: positional encoded image.
    """

    # Convert the image to grayscale if it's RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    encoded_image = np.zeros((image.shape[0] * steps, timeframe))
    width_resolution = 255.00 / steps

    #Gaussian Distribution's Probability Density Function (PDF)
    pdf = lambda mu, sigma, x: (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    helper = lambda mu, sigma, x: pdf(mu, sigma, x) / (1/(sigma * np.sqrt(2 * np.pi)))

    for i in range(image.shape[0]):
        for k in range(steps):
            index = (i * steps) + (k)
            mean = k * width_resolution
            time = int((1 - helper(mean, std, image[i])) * (timeframe-1))
            encoded_image[index, time] = 1
            #print(image[i], index, mean, time)
    return encoded_image

def poissonValues(image, steps=10, timeframe=20):
    #TODO: migrate to PyTorch
    """Assumes that spikes follow a Poisson distribution with and average firing of
        lambda_val (which is 1 / timeframe here). Then assuming we know k events have happend,
        and since events in a poisson distribution happen with an exponential distribution timeline, 
        we can use that to find the time that the k events are happening, using the parameters
        from the Poisson distribution. (k here being the firing rate of the neurons that we have fixed)
        NOTE: images are in grayscale.

    Args:
        image (np.ndarray): the input image.
        steps (int, optional): the number of classes we divide 0 to 255.0 into, Defaults to 10.
        timeframe (int, optional): the length which the spikes happen in, Defaults to 20.

    Returns:
        np.ndarray: poisson encoded image.
    """
    if timeframe < steps:
        raise ValueError("steps cannot be more than the timeframe.")
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    encoded_image = np.zeros((image.shape[0], timeframe))
    width_resolution = 255.00 / steps

    for i in range(image.shape[0]):
        amount_of_numbers = (image[i]//width_resolution)
        lambda_val = (1/timeframe)
        if amount_of_numbers > steps:
            amount_of_numbers = steps
        inter_arrival_times = set()
        while len(inter_arrival_times) < amount_of_numbers:
            random_number = np.random.exponential(scale=1/lambda_val)
            integer = int((random_number))
            if 0 <= integer < (timeframe):
                inter_arrival_times.add(integer)
        for timestamp in inter_arrival_times:
            encoded_image[i, int(timestamp)] = 1
            
    return encoded_image


"""if __name__ == "__main__":
    test_cases = ["bird", "bridge", "camera", "circles", "crosses", "goldhill1", "horiz", "lena1", "montage", "slope", "squares", "text"]
    colors = ((153, 0, 0), (255, 102, 102), (153, 76, 0), (255, 178, 102),
              (153, 153, 0), (255, 255, 102), (76, 153, 0), (178, 255, 102),
              (0, 153, 0), (102, 255, 102), (0, 153, 76), (102, 255, 178),
              (0, 153, 153), (102, 153, 153), (0, 76, 153), (102, 178, 255),
              (0, 0, 153), (102, 102, 255), (76, 0, 153), (178, 102, 255),
              (153, 0, 153), (255, 102, 255), (153, 0, 76), (255, 102, 178),
              (64, 64, 64), (192, 192, 192))
    
    name = "bird"
    for name in test_cases:
    #if True:
        timeframe = 20
        res = 32
        steps = 2
        #std = 0.01
        try:
            img = Image.open("./Photos/"+name+".tif")
        except IOError:
            print("Error: File not found or image cannot be opened. " + name)

        img = img.resize((res,res))
        width, height = img.size
        new_img_array = poissonValues(image=np.array(img).flatten(), timeframe=timeframe, steps=steps)
        
        #TODO: add colors to charts to be recognized better!
        #colors = np.random.uniform(0, 255, width*height)

        x_raster, y_raster = [], []
        for i in range(new_img_array.shape[0]):
            for j in range(timeframe):
                if new_img_array[i][j] == 1:
                    x_raster.append(j)
                    y_raster.append(i)
        x_raster, y_raster = np.array(x_raster), np.array(y_raster)
        plt.figure(figsize=(5,9))
        plt.scatter(x_raster, y_raster, c = "DARKGREEN", s=1, marker=".")
        plt.xlabel("time")
        plt.ylabel("neurons")
        plt.title("Encoding: Poisson, Testcase: " + name + "\n" + "max fire rate = " + str(steps) + ", res = " + str(res)+"*"+str(res))
        plt.ylim(-1, width*height)
        plt.xlim(-1, timeframe)
        plt.savefig("./Charts/raster_poisson_"+name+".jpg", dpi=600)
        plt.clf()

        temp_image = np.copy(np.array(img))
        for i in range(temp_image.shape[0]):
            for j in range(temp_image.shape[1]):
                temp_image[i, j] = np.floor(temp_image[i,j]/255.00*timeframe)*(255.00/timeframe)

        temp_image = Image.fromarray(temp_image)
        temp_image.save("./Charts/"+name+"_original_fulres.jpg", format="JPEG")

        new_image = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range (width):
            for j in range (height):
                for k in range(20):
                    if (new_img_array[(i*width)+j, k] == 1):
                        for m in range(3):
                            new_image[i, j, m] = colors[k][m]

        new_image = Image.fromarray(new_image, mode="RGB")
        new_image = new_image.convert('RGB')
        new_image.save("./Charts/new_"+name+".jpg", format="JPEG")"""
