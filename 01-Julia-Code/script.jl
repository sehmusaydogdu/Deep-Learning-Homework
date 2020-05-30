#Execirse 0
using DelimitedFiles, Statistics, Random
filePath = download("https://raw.githubusercontent.com/ilkerkesen/ufldl-tutorial/master/ex1/housing.data")

#Exercise 1
data = readdlm(filePath)

#Exercise 2
y_array = data[:,end] ## size= 506 x 1
y_array = reshape(y_array,(1,506)) ## size= 1 x 506

matrix = fill(0.0,(13,506))
for i in 1:13, j in 1:506
    matrix[i,j] = data[j,i]
end

#Exercise 3
mean_array = mean(matrix,dims=2)  
std_array = std(matrix;dims=2) 
for i in 1:13, j in 1:506
    matrix[i,j] = (matrix[i,j] - mean_array[i])/(std_array[i])
end

#Exercise 4
xtrn = fill(0.0,(13,400))
xtest = fill(0.0,(13,106))

ytrn = fill(0.0,(1,400))
ytest = fill(0.0,(1,106))

Random.seed!(1) 
randomValues = randperm(506) ##random generate value

xtrn = matrix[:,randomValues[1:400]]     ## xtrn filled
ytrn = y_array[:,randomValues[1:400]]    ## ytrn fiiled
xtest = matrix[:,randomValues[401:end]]  ## xtest filled
ytest = y_array[:,randomValues[401:end]] ## ytest filled

#Exercise 5
w = randn(1,13) * (0.1) #define weight

#Exercise 6
ypred_train = w * xtrn

#Exercise 7
total_train_loss = fill(0.0,(1,1))
train_loss = (ytrn - ypred_train).^2
for i in 1:400
    total_train_loss[1] = total_train_loss[1] + train_loss[i]
end
total_train_loss = (1/(2*400))*total_train_loss

ypred_test = w * xtest
total_test_loss = fill(0.0,(1,1))
test_loss = (ytest - ypred_test).^2
for i in 1:106
    total_test_loss[1] = total_test_loss[1] + test_loss[i]
end
total_test_loss = (1/(2*106))*total_test_loss

#Exercise 8
result = fill(0.0,(1,1))
yError = ytrn - ypred_train
ySqrt  = sqrt(total_train_loss) 
for i in 1:400
    result[1] = yError[i] < ySqrt[1]  ? result[1] + 1 : result[1]
end