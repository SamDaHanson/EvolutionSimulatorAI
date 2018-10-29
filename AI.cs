using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
//using TMPro;

public class AI : MonoBehaviour {

    //Evolution Simulator

    public SpawnCreature spawner;
    public Movement mover;
    public GameObject neuronSprite;
    public GameObject goal;
    public GameObject display;

    public float startingInputs = 0.5f;
    public float startingWeights = 0.5f;
    public int hidden1 = 10;
    public float bias1 = 0.25f;
    public int hidden2 = 10;
    public float bias2 = 0.25f;
    public float outBias = 0.2f;

    public int addConstants = 0;

    public float learningRate = 0.2f;
    public float muscleLearningRate = 0.001f;

    public void Update()
    {
        learningRate = learningRate;
        muscleLearningRate = muscleLearningRate;
        //display.transform.GetChild(0).GetComponent<TextMeshProUGUI>();
        //TextMeshPro texty = gameObject.transform.GetChild(2).GetChild(x).GetChild(0).GetComponent<TextMeshProUGUI>();
    }

    public GameObject neuron;
    public GameObject weight;

    public float[] inputLayer;
    public float[] outputLayer;
    public float[] yLayer;

    public float runningAverageCost = 0.0f;

    //For input improvement  
    public float[] muscleDer0;
    //Input = before then after

    private float[,] weightLayer0;
    private float[] biasLayer1;
    private float[] hiddenLayer1;
    private float[,] weightLayer1;
    private float[] biasLayer2;
    private float[] hiddenLayer2;
    private float[,] weightLayer2;
    private float[] outputBias;

    private int numberOfTrains = 0;
    private float hugeTotalCost = 0.0f;

    //For robust backprop (just to make sure it works)
    private float[] hiddenLayer1a;
    private float[] hiddenLayer2a;
    private float[] outputLayera;

    //Derivative Weights (used for gradient descent)
    private float[] derInputs;
    private float[,] derWeights0;
    private float[,] derWeights1;
    private float[,] derWeights2;
    private float[] derBias2;
    private float[] derBias1;
    private float[] derBias0;

    private float[] oldDers;   //To compare the net for a forward-backward network

    private float[] error;
    private float[] cost;

    //private List<> derWeights0;
    private bool zoomedIn = false;

    /*Functions
        SpawnNeuralNet()
        RunNetwork(bool goingBackwards)

        NumberfyNeurons(bool on)      //When zoomed in
        ZoomIn()

        ColorNeurons()
        ColorWeights(Transform weightLayer, float[,] weights)
        DrawWeights(Transform macroLayer, float[,] microLayer, int inLayer)
        DrawNeurons(Transform macroLayer, float[] microLayer, int numNeurons, int numRows = 1)
        
        sigmoid(float value)
        sigmoidDerivative(float value)
        BackPropagation(bool goingBackwards)
        GradientDescent()

        AssignMuscles(bool goingBackwards, float learningRate = 0.001f)

        RunOnce(bool goingBackwards)
        MoveMuscles()
    */

    public void SpawnNeuralNet()
    {
        muscleDer0 = new float[inputLayer.Length];
        int numMuscles = spawner.muscleList.Count;
        int numJoints = spawner.jointList.Count;

        //Inputs and Outputs of the Neural Network
        inputLayer = new float[numMuscles * 2 + numJoints + addConstants + 2];    //+1 for forwards/backwards
        muscleDer0 = new float[inputLayer.Length];                              //Used for muscle adjustment
        oldDers = new float[inputLayer.Length];
        outputLayer = new float[numJoints * 2];
        outputLayera = new float[numJoints * 2];
        yLayer = new float[outputLayer.Length];
        System.Random rnd = new System.Random();
        bool chooseRandom = true;
        for (int x = 0; x < inputLayer.Length; x++)
        {
            if (x < numMuscles * 2)
            {
                if (chooseRandom)
                {
                    //Randomizer
                    inputLayer[x] = 0.9f * (2 * (float)rnd.NextDouble() - 1) + 0.1f;
                }
                else
                {
                    if (x % 2 == 0)
                    {
                        inputLayer[x] = -0.5f;      //start
                    }
                    else
                    {
                        inputLayer[x] = 0.5f;       //end
                    }
                }
            }
            else if (x < numMuscles * 2 + numJoints)
            {
                //Debug.Log("Inputting Joint "+x);
                Vector3 goalDist = spawner.jointList[x - (numMuscles * 2)].transform.position - goal.transform.position;
                float scalarDist = Mathf.Sqrt(Mathf.Pow(goalDist.x, 2) + Mathf.Pow(goalDist.y, 2) + Mathf.Pow(goalDist.z, 2));
                //Debug.Log(scalarDist);
                inputLayer[x] = scalarDist;
            }
            else if (x < numMuscles * 2 + numJoints + addConstants)
            {
                //Debug.Log("Inputting Constants "+x);
                if (x%2==0)
                {
                    inputLayer[x - numMuscles * 2 - numJoints] = 1.0f;
                }
                else
                {
                    inputLayer[x - numMuscles * 2 - numJoints] = -1.0f;
                }
            }
            else
            {
                //Debug.Log("Inputting Forward/Backwards");
                if (x == inputLayer.Length-1)
                {
                    //Forwards-Backwards
                    inputLayer[x] = 0.0f;
                }
                else
                {
                    //NumJointOnGround
                    inputLayer[x] = spawner.NumJointsOnGround();
                }
            }
            //Debug.Log("x: "+x+" InLayer="+inputLayer[x]);
        }

        weightLayer0 = new float[hidden1, inputLayer.Length];
        for (int x = 0; x < weightLayer0.GetLength(0); x++)
        {
            for (int y = 0; y < weightLayer0.GetLength(1); y++)
            {
                weightLayer0[x, y] = 0.9f * (2 * (float)rnd.NextDouble() - 1.0f) + 0.1f;
            }
        }

        //Maybe not needed
        hiddenLayer1 = new float[hidden1 + addConstants];
        hiddenLayer1a = new float[hidden1 + addConstants];
        for (int x = 0; x < hiddenLayer1.Length; x++)
        {
            hiddenLayer1[x] = 0.5f;
        }
        for (int x = 0; x < addConstants; x++)
        {
            hiddenLayer1[hiddenLayer1.Length - 1 - x] = 1.0f;
        }
        biasLayer1 = new float[hidden1 + addConstants];
        for (int x = 0; x < biasLayer1.Length; x++)
        {
            biasLayer1[x] = bias1;
        }

        weightLayer1 = new float[hidden2, hidden1 + addConstants];
        //Debug.Log("WeightLayer1(0): "+weightLayer1.GetLength(0));
        //sDebug.Log("WeightLayer1(1): "+weightLayer1.GetLength(1));
        for (int x = 0; x < weightLayer1.GetLength(0); x++)
        {
            for (int y = 0; y < weightLayer1.GetLength(1); y++)
            {
                //weightLayer1[x,y] = startingWeights;
                weightLayer1[x, y] = 0.9f * (2 * (float)rnd.NextDouble() - 1.0f) + 0.1f;
            }
        }

        //Maybe not needed
        hiddenLayer2 = new float[hidden2 + addConstants];
        hiddenLayer2a = new float[hidden2 + addConstants];
        for (int x = 0; x < hiddenLayer2.Length; x++)
        {
            hiddenLayer2[x] = 0.5f;
        }
        for (int x = 0; x < addConstants; x++)
        {
            hiddenLayer2[hiddenLayer2.Length - 1 - x] = 1.0f;
        }

        biasLayer2 = new float[hidden2 + addConstants];
        for (int x = 0; x < biasLayer2.Length; x++)
        {
            biasLayer2[x] = bias2;
        }

        weightLayer2 = new float[outputLayer.Length, hidden2 + addConstants];
        for (int x = 0; x < weightLayer2.GetLength(0); x++)
        {
            for (int y = 0; y < weightLayer2.GetLength(1); y++)
            {
                //weightLayer2[x, y] = startingWeights;
                weightLayer2[x, y] = 0.9f * (2 * (float)rnd.NextDouble() - 1.0f) + 0.1f;
            }
        }

        for (int x = 0; x < outputLayer.Length; x++)
        {
            outputLayer[x] = 0.5f;
        }
        outputBias = new float[outputLayer.Length];
        for (int x = 0; x < outputBias.Length; x++)
        {
            outputBias[x] = outBias;
        }

        for (int x = 0; x < yLayer.Length; x++)
        {
            yLayer[x] = 0.5f;
        }

        DrawNeurons(gameObject.transform.GetChild(2).transform, inputLayer, inputLayer.Length);
        DrawNeurons(gameObject.transform.GetChild(3).transform, hiddenLayer1, hiddenLayer1.Length);
        DrawNeurons(gameObject.transform.GetChild(4).transform, hiddenLayer2, hiddenLayer2.Length);
        DrawNeurons(gameObject.transform.GetChild(5).transform, outputLayer, outputLayer.Length);
        DrawNeurons(gameObject.transform.GetChild(6).transform, yLayer, yLayer.Length);

        DrawWeights(gameObject.transform.GetChild(7).transform, weightLayer0, 2);
        DrawWeights(gameObject.transform.GetChild(8).transform, weightLayer1, 3);
        DrawWeights(gameObject.transform.GetChild(9).transform, weightLayer2, 4);

        ColorWeights(gameObject.transform.GetChild(7).transform, weightLayer0);
        ColorWeights(gameObject.transform.GetChild(8).transform, weightLayer1);
        ColorWeights(gameObject.transform.GetChild(9).transform, weightLayer2);

        ColorNeurons(gameObject.transform.GetChild(2).transform, inputLayer);
        ColorNeurons(gameObject.transform.GetChild(3).transform, hiddenLayer1);
        ColorNeurons(gameObject.transform.GetChild(4).transform, hiddenLayer2);
        ColorNeurons(gameObject.transform.GetChild(5).transform, outputLayer);
        ColorNeurons(gameObject.transform.GetChild(6).transform, yLayer);
    }

    private void RunNetwork(bool goingBackwards)
    {
        numberOfTrains++;
        //Joints V
        for (int x = 2 * spawner.muscleList.Count; x < inputLayer.Length - addConstants - 2; x++)
        {
            int score = x - 2 * spawner.muscleList.Count;
            float initialPos = mover.initialDistances[x - 2 * spawner.muscleList.Count];
            float endingPos = Mathf.Sqrt(Mathf.Pow(mover.endingDistance[x - 2 * spawner.muscleList.Count].x, 2) + Mathf.Pow(mover.endingDistance[x - 2 * spawner.muscleList.Count].y, 2) + Mathf.Pow(mover.endingDistance[x - 2 * spawner.muscleList.Count].z, 2));
            //float descalar = 0.01f;
            Vector3 goalDist = spawner.jointList[x - (spawner.muscleList.Count * 2)].transform.position - goal.transform.position;
            float dist = Mathf.Sqrt(Mathf.Pow(goalDist.x, 2) + Mathf.Pow(goalDist.y, 2) + Mathf.Pow(goalDist.z, 2)) / mover.initialDistances[x - (spawner.muscleList.Count * 2)];
            dist = sigmoid(1.0f - endingPos / initialPos);
            inputLayer[x] = dist;
            inputLayer[x] = inputLayer[x] * 2 - 1;
        }

        //***Input Bounce back from 1.0 and 0.0*** could create more learning
        for (int x = 0; x < inputLayer.Length; x++)
        {
            if (inputLayer[x] >= 0.99f)
            {
                inputLayer[x] = 0.5f;
            }
            else if (inputLayer[x] <= -0.99f)
            {
                inputLayer[x] = -0.5f;
            }
        }

        if (goingBackwards)
        {
            inputLayer[inputLayer.Length - 1] = -1.0f;
        }
        else
        {
            inputLayer[inputLayer.Length - 1] = 1.0f;
        }

        inputLayer[inputLayer.Length - 2] = spawner.NumJointsOnGround();

        for (int x = 0; x < hiddenLayer1.Length - addConstants; x++)
        {
            hiddenLayer1a[x] = 0.0f;
            for (int y = 0; y < inputLayer.Length; y++)
            {
                hiddenLayer1a[x] += inputLayer[y] * weightLayer0[x, y] + biasLayer1[x];
            }
            hiddenLayer1a[x] += biasLayer1[x];
            hiddenLayer1[x] = sigmoid(hiddenLayer1a[x]);
            hiddenLayer1[x] = hiddenLayer1[x] * 2 - 1;
        }

        for (int x = 0; x < hiddenLayer2.Length - addConstants; x++)
        {
            hiddenLayer2a[x] = 0.0f;
            for (int y = 0; y < hiddenLayer1.Length; y++)
            {
                hiddenLayer2a[x] += hiddenLayer1[y] * weightLayer1[x, y] + biasLayer2[x];
            }
            hiddenLayer2a[x] += biasLayer2[x];
            hiddenLayer2[x] = sigmoid(hiddenLayer2a[x]);
            hiddenLayer2[x] = hiddenLayer2[x] * 2 - 1;
        }

        for (int x = 0; x < outputLayer.Length; x++)
        {
            outputLayera[x] = 0.0f;
            for (int y = 0; y < hiddenLayer2.Length; y++)
            {
                outputLayera[x] += hiddenLayer2[y] * weightLayer2[x, y] + outputBias[x];
            }
            outputLayera[x] += outputBias[x];
            outputLayer[x] = sigmoid(outputLayera[x]);
            outputLayer[x] = outputLayer[x] * 2 - 1;
        }

        float totalCost = 0.0f;
        float sumChange = 0.0f;
        float sumTravel = 0.0f;
        error = new float[outputLayer.Length];
        for (int x = 0; x < outputLayer.Length; x++)
        {
            cost = new float[error.Length];
            if (x < outputLayer.Length / 2)
            {
                //First set is for the distance moved (we want to increase towards infinity)
                float realChange;
                if (mover.hasData)
                {
                    float starting = Mathf.Sqrt(Mathf.Pow(mover.startingDistance[x].x, 2) + Mathf.Pow(mover.startingDistance[x].y, 2) + Mathf.Pow(mover.startingDistance[x].z, 2));
                    float ending = Mathf.Sqrt(Mathf.Pow(mover.endingDistance[x].x, 2) + Mathf.Pow(mover.endingDistance[x].y, 2) + Mathf.Pow(mover.endingDistance[x].z, 2));
                    //goingBackwards = false;     //So that back is likely -, and forward is likely +
                    if (goingBackwards)
                    {
                        realChange = ending - starting;
                    }
                    else
                    {
                        realChange = starting - ending;
                    }
                    //realChange = starting - ending;
                    realChange = sigmoid(realChange);
                    realChange = realChange * 2 - 1;
                    yLayer[x] = realChange;
                    //Debug.Log("RealChange: " + realChange + " Ending: " + ending + " Starting: " + starting);
                    sumChange += realChange;
                }
                else
                {
                    realChange = 0.5f;
                }
                error[x] = outputLayer[x] - realChange;
            }
            else
            {
                //Second set is for remaining remaining distance (we want to decrease towards 0)
                float endingPosition;
                if (mover.hasData)
                {
                    float initialPos = mover.initialDistances[x - outputLayer.Length / 2];
                    float endingPos = Mathf.Sqrt(Mathf.Pow(mover.endingDistance[x - outputLayer.Length / 2].x, 2) + Mathf.Pow(mover.endingDistance[x - outputLayer.Length / 2].y, 2) + Mathf.Pow(mover.endingDistance[x - outputLayer.Length / 2].z, 2));
                    goingBackwards = false;
                    if (goingBackwards)
                    {
                        endingPosition = sigmoid(endingPos / initialPos - 1.0f);
                    }
                    else
                    {
                        endingPosition = sigmoid(1.0f - endingPos / initialPos);
                    }
                    //Debug.Log("Result: "+endingPosition+", EndingPos: "+endingPos+ ", InitalPos: "+initialPos);
                }
                else
                {
                    endingPosition = sigmoid(1.0f);
                }
                endingPosition = endingPosition * 2 - 1;
                yLayer[x] = endingPosition;
                error[x] = outputLayer[x] - endingPosition;
                sumTravel += endingPosition;
            }
            cost[x] = Mathf.Pow(error[x], 2);
            totalCost += cost[x];
        }
        hugeTotalCost += totalCost;
        //Debug.Log("Total Cost: " + totalCost);
        //Debug.Log("Average Total Cost: " + hugeTotalCost/numberOfTrains);
        float sumDistance = 0.0f;
        int n = 0;
        for (n = 0; n < mover.endingDistance.Length; n++)
        {
            sumDistance += Mathf.Sqrt(Mathf.Pow(mover.endingDistance[n].x, 2) + Mathf.Pow(mover.endingDistance[n].y, 2) + Mathf.Pow(mover.endingDistance[n].z, 2));
        }

        float avgChange = sumChange / n;
        //Debug.Log("Average Change: " + avgChange);
        float avgTravel = sumTravel / n;
        //Debug.Log("Average Travel: " + avgTravel);
        float avgDistance = sumDistance / n;
        //Debug.Log("Average Current Distance: " + avgDistance);

        //Color updated neurons
        ColorNeurons(gameObject.transform.GetChild(2).transform, inputLayer);
        ColorNeurons(gameObject.transform.GetChild(3).transform, hiddenLayer1);
        ColorNeurons(gameObject.transform.GetChild(4).transform, hiddenLayer2);
        ColorNeurons(gameObject.transform.GetChild(5).transform, outputLayer);
        ColorNeurons(gameObject.transform.GetChild(6).transform, yLayer);

        //TextMeshPro infoPanel = gameObject.transform.parent.GetChild(2).GetChild(0).GetChild(1).GetChild(0).GetComponent<TextMeshPro>();
        //infoPanel.setText("Test");
    }

    private void NumberfyNeurons(bool on)
    {
        /*
        for (int x = 0; x < inputLayer.Length; x++)
        {
            TextMeshPro texty = gameObject.transform.GetChild(2).GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if (on)
            {
                texty.SetText(inputLayer[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }
        }
        for (int x = 0; x < hiddenLayer1.Length; x++)
        {
            TextMeshPro texty = gameObject.transform.GetChild(3).GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if (on)
            {
                texty.SetText(hiddenLayer2[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }
        }
        for (int x = 0; x < hiddenLayer2.Length; x++)
        {
            TextMeshPro texty = gameObject.transform.GetChild(4).GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if (on)
            {
                texty.SetText(hiddenLayer2[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }
        }
        for (int x = 0; x < outputLayer.Length; x++)
        {
            TextMeshPro texty = gameObject.transform.GetChild(5).GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if (on)
            {
                texty.SetText(outputLayer[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }
        }
        for (int x = 0; x < yLayer.Length; x++)
        {
            TextMeshPro texty = gameObject.transform.GetChild(6).GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if (on)
            {
                texty.SetText(yLayer[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }
        }
        */
    }

    public void ZoomIn()
    {
        if (!zoomedIn)
        {
            gameObject.transform.localPosition = new Vector3(-7.5f, -1.5f, 6.0f);
            gameObject.transform.localScale = new Vector3(1.5f, 1.5f, 1.0f);
            zoomedIn = true;
            NumberfyNeurons(true);
        }
        else
        {
            gameObject.transform.localPosition = new Vector3(-2.85f, -3.25f, 6.0f);
            gameObject.transform.localScale = new Vector3(0.6f,0.6f,1.0f);
            zoomedIn = false;
            NumberfyNeurons(false);
        }
    }

    private void ColorNeurons(Transform neuronLayer, float[] neurons)
    {
        for (int x = 0; x < neurons.Length; x++)
        {
            float r,g,b;
            r = 1.0f;
            g = sigmoid(neurons[x]);
            //Debug.Log(g);
            b = 0.0f;
            neuronLayer.GetChild(x).GetComponent<SpriteRenderer>().color =  new Color(r,g,b);
            
            /*TextMeshPro texty = neuronLayer.GetChild(x).GetChild(0).GetComponent<TextMeshPro>();
            if(zoomedIn)
            {
                texty.SetText(neurons[x].ToString("F2"));
            }
            else
            {
                texty.SetText("");
            }*/
        }
    }

    private void ColorWeights(Transform weightLayer, float[,] weights)
    {
        int currentWeight = 0;
        for (int x = 0; x < weights.GetLength(1); x++)
        {
            for (int y = 0; y < weights.GetLength(0); y++)
            {
                float r,g,b,a;
                r = g = b = sigmoid(weights[y,x]);
                a = Mathf.Abs(2*(sigmoid((weights[y,x]))-0.5f));
                weightLayer.GetChild(currentWeight).GetComponent<SpriteRenderer>().color = new Color(r,g,b,a);
                currentWeight++;
            }
        } 
    }

    private void DrawWeights(Transform macroLayer, float[,] microLayer, int inLayer)
    {
        for(int x = 0; x < microLayer.GetLength(1); x++)
        {
            for(int y = 0; y < microLayer.GetLength(0); y++)
            {
                float verticalPosition = (gameObject.transform.GetChild(inLayer).transform.GetChild(x).localPosition.y + gameObject.transform.GetChild(inLayer+1).transform.GetChild(y).localPosition.y)/2;
                float angle = 360.0f*(Mathf.Atan(verticalPosition-gameObject.transform.GetChild(inLayer).transform.GetChild(x).localPosition.y)/1)/(2*Mathf.PI);
                float scale = 2*Mathf.Sqrt(Mathf.Pow(verticalPosition-gameObject.transform.GetChild(inLayer).transform.GetChild(x).localPosition.y,2) + 1);
                GameObject weightObj = Instantiate(weight, new Vector3(0.0f,verticalPosition,0.0f), macroLayer.rotation, macroLayer);
                weightObj.transform.Rotate(new Vector3(0, 0, angle));
                weightObj.transform.localPosition = new Vector3(0.0f,verticalPosition,0.0f);
                weightObj.transform.localScale = new Vector3(scale,0.1f,1.0f);
            }
        }
    }

    public void DrawNeurons(Transform macroLayer, float[] microLayer, int numNeurons, int numRows = 1)
    {
        float scale = 0.25f;
        float height = 3.0f;
        float positionMax = height - scale;
        float room = positionMax - height / 2;
        int remainderNeurons = numNeurons % numRows;
        scale = numRows*(room/(numNeurons));                //Might as well use scale again
        for (int row = 1; row <= numRows; row++)
        {
            int neuronsPerRow = numNeurons / numRows;
            if (remainderNeurons > 0)
            {
                neuronsPerRow++;
            }
            for (int x = 0; x < neuronsPerRow; x++)
            {
                float verticalPosition = 0;
                if (neuronsPerRow % 2 == 1)
                {
                    if (x == 0)
                    {
                        verticalPosition = 0;
                    }
                    else if (x % 2 == 1)
                    {
                        verticalPosition = (x + 1) * scale;
                    }
                    else
                    {
                        x = x - 1;
                        verticalPosition = (x + 1) * scale;
                        x++;
                    }
                }
                else
                {
                    if (x % 2 == 1)
                    {
                        x = x - 1;
                        verticalPosition = (x + 1) * scale;
                        x++;
                    }
                    else
                    {
                        verticalPosition = (x + 1) * scale;
                    }
                }
                neuron = Instantiate(neuronSprite, macroLayer.position, macroLayer.rotation, macroLayer);
                neuron.transform.localScale = new Vector3(0.25f, 0.25f, 0.25f);
                if (x % 2 == 1)
                {
                    verticalPosition = verticalPosition * -1;
                }
                macroLayer.transform.GetChild(x).transform.localPosition = new Vector3(0, verticalPosition, 0);
                macroLayer.transform.GetChild(x).GetComponent<SpriteRenderer>().sortingOrder = 1;
            }
        }
    }

    public float sigmoid(float value)
    {
        return 1.0f/(1.0f + Mathf.Pow((float)Math.E, -value));
    }

    public float sigmoidDerivative(float value)
    {
        float s = sigmoid(value);
        return s*(1.0f - s);
    }

    private void BackPropagation(bool goingBackwards)
    {
        float[] hiddenDerivative2 = new float[hiddenLayer2.Length];
        derWeights2 = new float[outputLayer.Length,hiddenLayer2.Length];
        derWeights1 = new float[hiddenLayer2.Length,hiddenLayer1.Length];
        derWeights0 = new float[hiddenLayer1.Length,inputLayer.Length];
        derInputs = new float[inputLayer.Length];

        derBias2 = new float[outputLayer.Length];
        derBias1 = new float[hiddenLayer2.Length];
        derBias0 = new float[hiddenLayer1.Length];

        float[] muscleDer2 = new float[hiddenLayer2.Length];
        float[] muscleDer1 = new float[hiddenLayer1.Length];

        for (int x = 0; x < outputLayer.Length; x++)
        {
            float costDerivative = 2.0f*error[x];
            //May want to check this ******* sigDer of sigDer to get z(L) in a(L)=sig(z(L))
            float sigDerivative = sigmoidDerivative(outputLayer[x]);
            //Debug.Log("sigDerivative: "+sigDerivative);
            derBias2[x] = sigDerivative * costDerivative;
            for (int y = 0; y < hiddenLayer2.Length; y++)
            {
                derWeights2[x,y] = hiddenLayer2[y]*sigDerivative*costDerivative;
                //Debug.Log(derWeights2[x,y]);
                    //cost derived by weight2 values
                hiddenDerivative2[y] += weightLayer2[x,y]*sigDerivative*costDerivative;
                    //cost derived by hidden2 neuron values

                goingBackwards = false;
                if (goingBackwards)
                {
                    if (x < outputLayer.Length/2)
                    {
                        //Deltas
                        muscleDer2[y] += weightLayer2[x, y] * sigDerivative;
                            //Want the muscles to try and move away from 0.5 -> 0.0
                    } else
                    {
                        //Arrows
                        muscleDer2[y] -= weightLayer2[x, y] * sigDerivative;
                            //Want the arrows to continue to try and minimize distance from goal -> 1.0
                    }
                } else
                {
                    muscleDer2[y] += weightLayer2[x, y] * sigDerivative;
                        //Want the arrows and deltas to improve
                            //Arrows -> Goal (1.0)
                            //Deltas -> Forward Movement (1.0)
                }
            }
        }
        float[] hiddenDerivative1 = new float[hiddenLayer1.Length];
        for (int y = 0; y < hiddenLayer2.Length-addConstants; y++)
        {
            //Debug.Log(hiddenLayer2[y]);
            float sigDerivative2 = sigmoidDerivative(hiddenLayer2[y]);
            derBias1[y] = sigDerivative2 * hiddenDerivative2[y];
            for (int z = 0; z < hiddenLayer1.Length; z++)
            {
                //Debug.Log(hiddenDerivative2[y]);
                derWeights1[y,z] = hiddenLayer1[z]*sigDerivative2*hiddenDerivative2[y];
                hiddenDerivative1[z] += weightLayer1[y, z] * sigDerivative2 * hiddenDerivative2[y];
                muscleDer1[z] += weightLayer1[y,z] * sigDerivative2 * muscleDer2[y];
                //Debug.Log(muscleDer1[z]);
            }
        }

        for (int z = 0; z < hiddenLayer1.Length-addConstants; z++)
        {
            float sigDerivative3 = sigmoidDerivative(hiddenLayer1[z]);
            //Debug.Log(hiddenLayer1a[z]);
            //Debug.Log("sigDerivative3: "+sigDerivative3);     //Make sure != 0
            derBias0[z] = sigDerivative3 * hiddenDerivative1[z];
            for (int a = 0; a < inputLayer.Length; a++)
            {
                //Debug.Log(hiddenDerivative1[z]);
                derWeights0[z,a] = inputLayer[a]*sigDerivative3*hiddenDerivative1[z];
                //Debug.Log(derWeights0[z,a]);
                derInputs[a] += weightLayer0[z,a]*sigDerivative3*hiddenDerivative1[z];
                //Debug.Log("DerWeights0: "+derWeights0[z,a]);
                muscleDer0[a] += weightLayer0[z, a] * sigDerivative3 * muscleDer1[z];
                //Debug.Log(muscleDer0[a]);
            }
        }
    }

    private void GradientDescent()
    {
        for (int x = 0; x < outputLayer.Length; x++)
        {
            for (int y = 0; y < hiddenLayer2.Length; y++)
            {
                //Debug.Log("Weight: "+weightLayer2[x,y]+" derWeights0: "+derWeights2[x,y]);
                weightLayer2[x,y] = weightLayer2[x,y]-derWeights2[x,y]*learningRate;
                outputBias[x] = outputBias[x]-derBias2[x]*learningRate;
            }
        }
        for (int x = 0; x < hiddenLayer2.Length-addConstants; x++)
        {
            for (int y = 0; y < hiddenLayer1.Length; y++)
            {
                weightLayer1[x,y] = weightLayer1[x,y]-derWeights1[x,y]*learningRate;
                biasLayer2[x] = biasLayer2[x]-derBias1[x]*learningRate;
            }
        }
        for (int x = 0; x < hiddenLayer1.Length-addConstants; x++)
        {
            for (int y = 0; y < inputLayer.Length; y++)
            {
                //Debug.Log("derWeights0("+x+","+y+"): "+ derWeights0[x,y]);
                //Debug.Log("weightLayer0("+x+","+y+"): "+ weightLayer0[x,y]);
                weightLayer0[x,y] = weightLayer0[x,y]-derWeights0[x,y]*learningRate;
                biasLayer1[x] = biasLayer1[x]-derBias0[x]*learningRate;
            }
        }
    }

    private void AssignMuscles(bool goingBackwards, float learningRate = 0.001f)
    {
        int negDer = 0;
        int posDer = 0;
        int zeroDer = 0;
        for (int x = 0; x < inputLayer.Length-spawner.jointList.Count-addConstants-2; x++)
        {
            //inputLayer[x] = inputLayer[x] + muscleDer0[x] * learningRate;
            goingBackwards = false;
            float derivative = muscleDer0[x];
            if (goingBackwards)
            {
                if (inputLayer[x] < 1)
                {
                    inputLayer[x] = inputLayer[x] + derivative * learningRate;
                }
                else
                {
                    inputLayer[x] = inputLayer[x] - derivative * learningRate;
                }
            } else
            {
                if (inputLayer[x] < 1)
                {
                    inputLayer[x] = inputLayer[x] + derivative * learningRate;
                }
                else
                {
                    inputLayer[x] = inputLayer[x] + derivative * learningRate;
                }
            }
            if (derivative > 0)
            {
                posDer++;
            }
            else if (derivative < 0)
            {
                negDer++;
            }
            else
            {
                zeroDer++;
            }

            if (inputLayer[x] > 1.0f)
            {
                inputLayer[x] = 1.0f;
            } else if(inputLayer[x] < -1.0f)
            {
                inputLayer[x] = -1.0f;

            }
        }
        /*
        Debug.Log("NegDer: " + negDer + ", PosDer: " + posDer +", ZeroDer: " + zeroDer);
        if (negDer > 0 && posDer > 0)
            Debug.Log("~~~~~~~~~~Changing Muscles~~~~~~~~~~");
        */
        ColorNeurons(gameObject.transform.GetChild(2).transform, inputLayer);
    }

    public void RunOnce(bool goingBackwards)
    {
        for (int x = 0; x < 1; x++)
        {
            //Run for results
            RunNetwork(goingBackwards);
            //Run Back-Propagation to find the changes we want
            BackPropagation(goingBackwards);
            //Run Gradient Descent to implent thoughtful changes
            GradientDescent();
            ColorWeights(gameObject.transform.GetChild(7).transform, weightLayer0);
            ColorWeights(gameObject.transform.GetChild(8).transform, weightLayer1);
            ColorWeights(gameObject.transform.GetChild(9).transform, weightLayer2);
        }
        //Assign new values as inputs to learn from
        AssignMuscles(goingBackwards, muscleLearningRate);
    }

    public void MoveMuscles()
    {
        spawner.MoveMuscles(inputLayer, addConstants);
    }
}
