import torch

torch.manual_seed(1)
# This is how you initialize random tensors of a specified set of dimensions
# In NLP, this 3x5 matrix could potentially represent a word.  
item = torch.randn(3,5)
print(item)

# A second item. 
item_2 = torch.randn(3, 5)

# Concats rows by default. For NLP, this is one way to represent 
# multiple words in a phrase or sentence -- by concatenating 
# tensors that represent words in a sequence.
item_plus_item_2 = torch.cat([item, item_2])
print(item_plus_item_2)


x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have ``requires_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)

# Now z has the computation history that relates itself to x and y
# Can we just take its values, and **detach** it from its history?
new_z = z.detach()

# ... does new_z have information to backprop to x and y?
# NO!
print(new_z.grad_fn)
# And how could it? ``z.detach()`` returns a tensor that shares the same storage
# as ``z``, but with the computation history forgotten. It doesn't know anything
# about how it was computed.
# In essence, we have broken the Tensor away from its past history


