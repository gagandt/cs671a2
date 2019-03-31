from cus_load import data
dataset = data("output")
x_train, y_leng, y_widt, y_colo, y_angl = dataset.load()
no_of_batches = 750
x_train = np.split(x_train, no_of_batches)
y_leng = np.split(y_leng, no_of_batches)
y_widt = np.split(y_widt, no_of_batches)
y_colo = np.split(y_colo, no_of_batches)
y_angl = np.split(y_angl, no_of_batches)


np.save()