import tensorflow as tf
import matplotlib.pyplot as plt


x = tf.linspace(-10.0, 10.0, 200+1)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.sigmoid(x)
    
dy_dx = tape.gradient(y, x)

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')
plt.show()

# Create two input variables
x1 = tf.linspace(-10.0, 10.0, 200+1)
x2 = tf.linspace(-5.0, 5.0, 200+1)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([x1, x2])
    # Create a more complex function with both variables
    y = tf.nn.sigmoid(x1) * tf.cos(x2)
    
# Get partial derivatives with respect to both variables    
dy_dx1 = tape.gradient(y, x1)
dy_dx2 = tape.gradient(y, x2)

# Plot original function and both partial derivatives
plt.figure(figsize=(10,6))
plt.plot(x1, y, label='y')
plt.plot(x1, dy_dx1, label='dy/dx1')
plt.plot(x1, dy_dx2, label='dy/dx2')
plt.legend()
plt.xlabel('x')
plt.title('Function and its partial derivatives')
plt.show()

# Clean up the tape since we used persistent=True
del tape


