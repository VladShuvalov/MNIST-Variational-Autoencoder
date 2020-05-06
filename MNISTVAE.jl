using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)

function factorized_gaussian_log_density(mu, logsig,xs)
  """
  logsig used to allow
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  return - log1pexp.(-b .* logit_means)
end

# sample from Diagonal Gaussian x ~ N(μ,σI) (using reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)

# sample from Bernoulli
sample_bernoulli(θ) = rand.(Bernoulli.(θ))

# Load MNIST data, binarise it, split into train and test sets (10000 each)
function load_binarized_mnist(train_size=10000, test_size=10000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

# Partition train into mini-batches of M=100
function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end

batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size))

## Load the Data
train_data, test_data = load_binarized_mnist();
train_x, train_label = train_data;
test_x, test_label = test_data;

## Model dimensions
#latent dimensionality=2 and number of hidden units=500, MNIST images are 28^2 pixels large
#Prior set as N(0,1)

Dz, Dh = 2, 500
Ddata = 28^2

log_prior(z) = sum(factorized_gaussian_log_density(0., 0., z))

function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z) """
  θ = decoder(z)
  return  sum(bernoulli_log_density(θ,x))
end

function joint_log_density(x,z)
  """ Compute joint log likelihood log_p(x,z) """
  return log_likelihood(x,z) .+ log_prior(z)
end

## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end

decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, 2*Dz), unpack_gaussian_params)

log_q(q_μ, q_logσ, z) = sum(factorized_gaussian_log_density(q_μ, q_logσ, z))

function elbo(x)
  """Scalar value, mean variational evidence lower bound over batch"""
  q_μ, q_logσ = encoder(x)
  z = sample_diag_gaussian(q_μ, q_logσ)
  joint_ll = log_likelihood(x,z)
  log_q_z = log_q(q_μ, q_logσ, z)
  elbo_estimate = sum(log_prior(z) + joint_ll - log_q_z) / 100
  return elbo_estimate
end

function loss(x)
  """scalar value for the variational loss over elements in the batch"""
  return -elbo(x)
end

## Training with gradient optimization:

function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params
  ps = Flux.params(encoder,decoder) #TODO parameters to update with gradient descent
  # ADAM optimizer with default parameters
  opt = ADAM(0.00001)
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x)
      gs = Flux.gradient(ps) do
        batch_loss = loss(d)
        return batch_loss
      end# compute gradients with respect to variational loss over batch
      Flux.Optimise.update!(opt,ps,gs)#TODO update the paramters with gradients
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end


# Train the model
train_model_params!(loss,encoder,decoder,train_x,test_x, nepochs=200)

### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_params.bson") encoder
@save joinpath(save_dir,"decoder_params.bson") decoder
@info "Saved model params in $save_dir"



## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"

## Visualization
using Images
using Plots
# make vector of digits into images, works on batches also

mnist_img(x) = ndims(x)==2 ? Gray.(permutedims(reshape(x,28,28,:), [2, 1, 3])) : Gray.(transpose(reshape(x,28,28)))

#Plots of 10 images generated from individual samples
#Generated images on top binary images sampled from MNIST below
using Plots.PlotMeasures
img_plots = Any[]
gen_plots = Any[]
for i in 1:10
  n = rand(1:1000)
  push!(gen_plots, plot(mnist_img_alt(sigmoid.(decoder(encoder(train_x)[1])))[:,:,n]))
  push!(img_plots, plot(mnist_img_alt(train_x[:,n])))
end

gr(display_type=:inline)
comparison_plots = vcat(gen_plots,img_plots)
Plots.plot(comparison_plots..., layout = (2, 10), axis=nothing)

#Plot of latent space
zs = encoder(train_x)[1]
scatter(zs[1,:], zs[2,:], group = train_label)
plot!(title = "2D Latent Space Of Fitted MNIST Data")

function interp(za,zb,alpha)
  return alpha * za .+ (1-alpha) * zb
end

plot_array = Any[]
for a in 0:0.1:0.9
  push!(plot_array, plot(mnist_img_alt(sigmoid.(decoder(interp(encoder(train_x[:,567])[1], encoder(train_x[:,312])[1], a))[:,1]))))
end
