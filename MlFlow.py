import mlflow
import mlflow.tensorflow
import tensorflow as tf

# 1. Enable MLflow autologging BEFORE defining or fitting the model.
# This logs metrics, parameters, and the final model automatically.
mlflow.tensorflow.autolog() 
# OR use the generic mlflow.autolog()

# Start an explicit MLflow run (optional but recommended for structure)
with mlflow.start_run() as run:
    # --- Model Definition (Your CNN Architecture) ---
    model = create_cnn_model() 
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 2. Fit the model. Metrics are logged automatically per epoch.
    model.fit(
        train_data, 
        train_labels, 
        epochs=10, 
        validation_data=(val_data, val_labels)
    )
    
    # 3. Access the tracking ID if needed
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

# To view results: Run 'mlflow ui' in your terminal and open http://localhost:5000