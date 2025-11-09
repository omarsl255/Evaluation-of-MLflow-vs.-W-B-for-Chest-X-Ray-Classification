import wandb
from wandb.keras import WandbMetricsLogger
import tensorflow as tf

# Define Hyperparameters (Best practice for W&B)
config = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "epochs": 10,
    "batch_size": 32,
    "dropout_1": 0.25 
}

# 1. Initialize W&B Run (connects to the W&B cloud dashboard)
run = wandb.init(
    project="Chest-XRay-Classification", 
    config=config
)

# --- Model Definition (Your CNN Architecture) ---
model = create_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 2. Fit the model, passing the W&B callback
model.fit(
    train_data, 
    train_labels, 
    epochs=config['epochs'],
    validation_data=(val_data, val_labels),
    # Use WandbMetricsLogger for standard metrics
    callbacks=[WandbMetricsLogger()] 
)

# 3. Finish the run
run.finish()

# To view results: A link to the W&B dashboard is printed in the terminal