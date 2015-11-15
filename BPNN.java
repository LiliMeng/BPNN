import java.io.*;

public class BPNN {
	private double learningRate, momentum;
	private double w[][] = new double[9][9]; //weight
	private double delta[][] = new double[9][9];
	private double u[] = new double[9]; 
	private double c[] = new double[1];
	private double s[] = new double[9];
	private double e[] = new double[9]; 
	
	public BPNN() {
		
		u[0] = 1.0; u[3] = 1.0; //bias term
		learningRate = 0.5;
		momentum = 0.9; 
		weightInitialization();
	}
	
	public static void main(String args[]) throws Exception {
		
		BPNN neuron = new BPNN();
		
		for(int i=0; i<10000; i++) {
			double E[] = {0.0, 0.0, 0.0, 0.0};
			double TE =0.0;
			
			neuron.u[1] = 1.0;
			neuron.u[2] = 1.0;
			neuron.c[0] = -1.0;
			neuron.output(neuron.u);
			neuron.backpropagation(neuron.c);
			E[0] = Math.pow(neuron.c[0]-neuron.u[8], 2); //error
			neuron.train();
			
			neuron.u[1] = -1000000.0;
			neuron.u[2] = -1000000.0;
			neuron.c[0] = -1.0;
			neuron.output(neuron.u);// a forward and backward iteration
			neuron.backpropagation(neuron.c);
			E[1] = Math.pow(neuron.c[0] - neuron.u[8], 2);
			neuron.train (); //update weight
			neuron.u[1] = 1.0;
			
			neuron.u[2] = -1000000.0;
			neuron.c[0] = 1.0;
			neuron.output(neuron.u);// a forward and backward iteration
			neuron.backpropagation(neuron.c);
			E[2] = Math.pow(neuron.c[0] - neuron.u[8], 2);
			neuron.train ();
				
			neuron.u[1] = -1000000.0;
			neuron.u[2] = 1.0;
			neuron.c[0] = 1.0;
			neuron.output(neuron.u);// a forward and backward iteration
			neuron.backpropagation(neuron.c);
			E[3] = Math.pow(neuron.c[0] - neuron.u[8], 2);
			neuron.train ();

			for (int j = 0; j < 4; j++) {
				TE = TE + 0.5 * E[j];
			}
			System.out.println(i + " " + TE);
			if (TE < 0.05)
				break;
			}
		}
		
		private double sigmoid(double si) {
			return (1.0 / (1.0 + Math.pow(Math.E, -si)));
		}

		private double customSigmoid(double si) {
			final double abound = -1.0;
			final double bbound = 1.0;
			double r = 0.0;
			double x = -1.2 + 2.4 / (1.0 + Math.pow(Math.E, -si));
			if (x <= abound)
				r = -1.0;
			if (x >= bbound)
				r = 1.0;
			if (x > abound && x <bbound)
				r = x;
			return (r);
		}

		private void weightInitialization() {//initialize weight
			for (int i = 4; i< 8; i++) {// initialize weight between input and hidden layer
				for (int j = 0; j < 3; j++) {
					while (w[i][j] == 0.0) {// ?
						w[i][j] = -2.0 + 4.0 * Math.random();
					}
				}
			}
			for (int j = 3; j < 8; j++) {
				while (w[8][j] == 0.0) {
					w[8][j] = -2.0 + 4.0 * Math.random();
				}
			}
		}

		public double output(double u[]) {// forward propagation step
			for (int i = 4; i< 8; i++) {
				s[i] = 0.0;
				for (int j = 0; j < 3; j++) {
					s[i] = s[i] + w[i][j] * u[j];
				}
				u[i] = customSigmoid(s[i]);
			}

			s[8] = 0.0;
			for (int j = 3; j < 8; j++) {
				s[8] = s[8] + w[8][j] * u[j];
			}
			u[8] = customSigmoid(s[8]);
			
			return (u[8]);
		}
		
		public void backpropagation(double c[]){
			e[8] = (c[0] - u[8]) * ((1/2.4) * (1.2 + u[8]) * (1.2-u[8]));
			for (int i = 4; i< 8; i++) {
				e[i] = ((1/2.4) * (1.2 + u[i]) * (1.2-u[i])) * w[8][i] * e[8];
			}
		}

		public void train () {
			for (int i = 4; i< 8; i++) {
				for (int j = 0; j < 3; j++) {
					w[i][j] = w[i][j] + learningRate * e[i] * u[j] + momentum * delta[i][j];
					delta[i][j] = learningRate * e[i] * u[j] + momentum * delta[i][j];
				}
			}
			for (int j = 3; j < 8; j++) {
				w[8][j] = w[8][j] + learningRate * e[8] * u[j] + momentum * delta[8][j];
				delta[8][j] = learningRate * e[8] * u[j] + momentum * delta[8][j];
			}
		}
		
	}

