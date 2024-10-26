import pickle
import numpy as np
import tensorflow as tf
import deepxde as dde

# Step 1: Load the model from the pickle file
with open("_PINN.pickle", "rb") as f:
    model_data = pickle.load(f)

# Inspect the loaded data
print("Loaded model data type:", type(model_data))
print("Loaded model data content:", model_data)

# Check if the loaded data is a list and inspect its contents
if isinstance(model_data, list):
    for i, item in enumerate(model_data):
        print(f"Item {i}: Type: {type(item)}, Value: {item}")

# Depending on what these items are, you might need to recreate the model
# If the list contains parameters, you will need to know how to rebuild the model
# Below is an example if you have to re-initialize a model


# This is an example, adjust it according to your model's architecture
# Rebuilding the model based on what you have
def setup_diffusion_sorption(filename, seed):
    # Your previous setup code goes here
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    D = 5e-4

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc_d = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, X):
        # compute u_t
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_d2 = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    # Rebuild the dataset, you might need to adjust based on what you stored
    # Assuming 'filename' and 'seed' are still valid
    dataset = PINNDatasetDiffSorption(filename, seed)

    # Note: you may need to adjust this based on your saved model parameters
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    # Initialize your model object
    model = dde.Model(data, net)

    return model


# Now, if the items in the list are relevant, use them accordingly
# For example, if the first item is the model parameters:
# model.set_weights(model_data[0])  # Example: if you need to set weights

# After rebuilding the model or extracting weights, make predictions
# Prepare input data for predictions
num_points = 100  # Number of points you want to predict
time_points = np.linspace(0, 500, num_points)  # Adjust based on your time domain
position_points = np.linspace(0, 1, num_points)  # Adjust based on your spatial domain

# Create a meshgrid for input
inputs = np.array(np.meshgrid(position_points, time_points)).T.reshape(-1, 2)

# Step 3: Make predictions if model is now a valid DeepXDE model object
if hasattr(model, "predict"):
    predictions = model.predict(inputs)

    # Step 4: Post-process the predictions (if needed)
    if isinstance(predictions, np.ndarray):
        predictions = predictions.reshape(num_points, num_points)  # Example reshaping

        # Optional: Save or plot predictions
        import matplotlib.pyplot as plt

        plt.imshow(predictions, extent=(0, 500, 0, 1), origin="lower", aspect="auto")
        plt.colorbar(label="Predicted Concentration")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Diffusion and Sorption Predictions")
        plt.show()
else:
    print("The reconstructed model is not a valid DeepXDE model object.")
