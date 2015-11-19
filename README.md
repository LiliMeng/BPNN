# BPNN
Backpropagation Neural Network by java

Neural Network Backpropagation

BPNN.java is for binary sigmoid
BPNN1.java is for bipolar sigmoid



/bipolar Sigmoid. Don't forget to change binary derivative f(x)*(1-f(x))  to bipolar derivative (f(x)+1)*0.5*(1-f(x))!
w[i][j] = 2* Math.random()-1; //Initialize the bias units with a random number between -1 and 1 in bipolarSigmoid case as -0.5~0.5 cannot converge

Implement a multi-layer perceptron and train it using the error-backpropagation algorithm.

For details:

Backpropagation: http://courses.ece.ubc.ca/592/PDFfiles/Backpropagation_c.pdf Assignment: https://courses.ece.ubc.ca/592/EECE592_WebSite_2009/Coursework_files/EE592-Robocode-Project-2015.pdf
