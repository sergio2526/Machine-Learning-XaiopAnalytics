using Microsoft.ML;
using Microsoft.ML.Data;
using System;

#pragma warning disable CS0649

namespace myMLApp
{
    class Program
    {

        public class IrisData
        {
            [LoadColumn(0)]
            public float Ropa;

            [LoadColumn(1)]
            public float Deportivo;

            [LoadColumn(2)]
            public float ropaInvierno;

            [LoadColumn(3)]
            public float ropaVerano;

            [LoadColumn(4)]
            public float ropaCaballero;

            [LoadColumn(5)]
            public float ropaDama;

            [LoadColumn(6)]
            public string Label;
        }

        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }


        public void general(){

            
                  bool cicle = true;
            while (cicle)
            {

                 Console.WriteLine("Ingrese valor #Ropa");
                var option1 = Console.ReadLine();
                float valueUser1 = Convert.ToInt32(option1);

                Console.WriteLine("Ingrese valor #Deportivo");
                var option2 = Console.ReadLine();
                float valueUser2 = Convert.ToInt32(option2);

                Console.WriteLine("Ingrese valor #ropaInvierno");
                var option3 = Console.ReadLine();
                 float valueUser3 = Convert.ToInt32(option3);

                Console.WriteLine("Ingrese valor #ropaVereno");
                var option4 = Console.ReadLine();
                
               float valueUser4 = Convert.ToInt32(option4);

                Console.WriteLine("Ingrese valor #ropaCaballero");
                var option5 = Console.ReadLine();
                
               float valueUser5 = Convert.ToInt32(option5);

                Console.WriteLine("Ingrese valor #ropaDama");
                var option6 = Console.ReadLine();
                
               float valueUser6 = Convert.ToInt32(option6);


            MLContext mlContext = new MLContext();


            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(path: "database.txt", hasHeader: true, separatorChar: ',');


            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Ropa", "Deportivo","ropaInvierno", "ropaVerano", "ropaCaballero","ropaDama"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingDataView);

            var prediction = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model).Predict(


                new IrisData()
                {


                    Ropa = valueUser1,
                    Deportivo = valueUser2,
                    ropaInvierno = valueUser3,
                    ropaVerano = valueUser4,
                    ropaCaballero =valueUser5,
                    ropaDama = valueUser6,
                    

                });

            Console.WriteLine($"El competidor mas relevante es : {prediction.PredictedLabels}");

            Console.WriteLine("Gracias por utilizar Xaiop Analytics");
            Console.ReadLine();
        }
        }

        static void Main(string[] args)
        {
            Program a  = new Program();

         a.general();
         

    }

    }
}