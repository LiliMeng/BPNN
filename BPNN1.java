//bipolar Sigmoid. Don't forget to change binary derivative f(x)*(1-f(x))  to bipolar derivative (f(x)+1)*0.5*(1-f(x))!

import java.io.FileWriter;  

public class BPNN 
{
	private double learningRate, momentum;
	private double w[][] = new double[9][9]; //weight
	private double delta[][] = new double[9][9];
	private double u[] = new double[9]; 
	private double desiredOutput[] = new double[1]; //desired feedForward value
	private double s[] = new double[9];
	private double e[] = new double[9]; 
	
	private static final String LINE_SEPARATOR = System.getProperty("line.separator"); 
	
	public BPNN() 
	{
		u[0] = 1.0; u[3] = 1.0; //bias term
		learningRate = 0.2;
		momentum = 0.0;
		weightInitialization();
	}
	   /**
		 * Return a binary binarySigmoid of the input X
		 * @param x The input
		 * @return f(x) = 1 / (1+e(-x)) 
		 */
	public double binarySigmoid(double x)
	{
		return 1/(1+Math.exp(-x));
	}
		
		
	public double bipolarSigmoid(double x)
	{	
		return -1+2/(1+Math.exp(-x));	
	}

	private void weightInitialization()
	{
		for (int i =4; i < 8; i++)
		{
			for(int j=0; j<3; j++)
			{
				while(w[i][j]==0.0)
				{
					w[i][j] = 2* Math.random()-1; //Initialize the bias units with a random number between -1 and 1 in bipolarSigmoid case as -0.5~0.5 cannot converge
				}
			}
		}
			
		for(int j = 3; j<8; j++)
		{
			while(w[8][j]==0.0)
			{
				w[8][j] =2*Math.random()-1; //Initialize the bias units with a random number between -1 and 1 in bipolarSigmoid case as -0.5~0.5 cannot converge
				//System.out.println(" " + w[8][j]);
			}
		}
			
	}
		
	public double feedForward()
	{
		for(int i=4; i<8; i++)
		{
			s[i]=0.0;
			
			for(int j=0; j<3; j++)
			{
				s[i]=s[i]+w[i][j]*u[j];
			}
				
			u[i] = bipolarSigmoid(s[i]);
				//System.out.println(u[i]);
		}

		s[8] = 0.0;
			
		for (int j = 3; j < 8; j++) 
		{
			s[8] = s[8] + w[8][j] * u[j];
		}
			
		u[8] = bipolarSigmoid(s[8]);
		//	System.out.println("u[8]"+u[8]);
			
		return u[8];
	}
		
	public void backpropagationError()
	{
		e[8] = (u[8]+1)*0.5*(1-u[8]) * (desiredOutput[0]-u[8]);
			
		for (int i = 4; i < 8; i++) 
		{
			e[i] = (u[i]+1)*0.5*(1-u[i])* w[8][i] * e[8];
		}
	}

	public void updateWeights () 
	{
		for (int i = 4; i< 8; i++) 
		{
			for (int j = 0; j < 3; j++)
			{
				w[i][j] = w[i][j] + learningRate * e[i] * u[j] + momentum * delta[i][j];
				delta[i][j] = learningRate * e[i] * u[j] + momentum * delta[i][j];
			}
		}
		for (int j = 3; j < 8; j++) 
		{
			w[8][j] = w[8][j] + learningRate * e[8] * u[j] + momentum * delta[8][j];
			delta[8][j] = learningRate * e[8] * u[j] + momentum * delta[8][j];
		}
	}
		
	public static void main(String args[]) throws Exception 
	{
			
			BPNN neuron = new BPNN();
	        
		    FileWriter fw = new FileWriter("/home/lili/workspace/EECE592/BPNN/src/result.txt",true);  
		   
			
			for(int i=0; i<1000; i++) 
			{
				
				double error[] = {0.0, 0.0, 0.0, 0.0};
				double totalError =0.0;
				
				neuron.u[1] = 1.0;
				neuron.u[2] = 1.0;
				neuron.desiredOutput[0] = -1.0;
				neuron.feedForward();
				neuron.backpropagationError();
				neuron.updateWeights();
				error[0] = Math.pow(neuron.desiredOutput[0]-neuron.u[8], 2); //error
				
				neuron.u[1] = -1;
				neuron.u[2] = -1;
				neuron.desiredOutput[0] = -1;
				neuron.feedForward();// a forward and backward iteration
				neuron.backpropagationError();
				neuron.updateWeights(); //update weight
				error[1] = Math.pow(neuron.desiredOutput[0] - neuron.u[8], 2);
				
				neuron.u[1] = -1;
				neuron.u[2] = 1;
				neuron.desiredOutput[0] = 1;
				neuron.feedForward();// a forward and backward iteration
				neuron.backpropagationError();
				neuron.updateWeights();
				error[2] = Math.pow(neuron.desiredOutput[0] - neuron.u[8], 2);
					
				neuron.u[1] = 1;
				neuron.u[2] = -1;
				neuron.desiredOutput[0] = 1;
				neuron.feedForward();// a forward and backward iteration
				neuron.backpropagationError();
				neuron.updateWeights();
				error[3] = Math.pow(neuron.desiredOutput[0] - neuron.u[8], 2);
				
				for (int j = 0; j < 4; j++) 
				{
					totalError = totalError + 0.5 * error[j];
				}
				
				System.out.println(i + " " + totalError);
				

				if (totalError < 0.05)
					break;
			   
			    fw.write(i+LINE_SEPARATOR+totalError);
			    
			   }
			   fw.flush();
			   fw.close();
		
		}
	}
