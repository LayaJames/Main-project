from flax.training import train_state
import numpy as np

# Initialize model parameters
params = model.init(jax.random.PRNGKey(0), train_inputs, train_masks)["params"]

# Create training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Training loop
batch_size = 32
num_epochs = 3

for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_masks = train_masks[i:i+batch_size]
        batch_labels = jnp.array(train_labels[i:i+batch_size])
        
        state.params, state.tx, loss = train_step(
            state.params, state.tx, batch_inputs, batch_masks, batch_labels
        )
        
    print(f"Epoch {epoch + 1}, Loss: {loss}")
