import argparse
import sys
from utilties import predict, chek_image_path, chek_model_path,red_json_file
import tensorflow as tf
import numpy as np



parser = argparse.ArgumentParser(
    description='deeplearning classifier to classify flowers and rosses.',
)
def main():
    parser.add_argument("image" , metavar='image', type=str, help='a list of images are avaliable in the test_images folder pick one ')
    parser.add_argument("model" , metavar='model', type=str, help='you got list of compiled modles in the directory models pic one ')
    parser.add_argument("--top_k" , metavar='top_k', type=int, help='the top number of probabilities')
    parser.add_argument("--category_names" , metavar='category_names', type=str, help='the top number of probabilities')
    args = parser.parse_args()
    prediction=None
    print(type(args))


    if (len(sys.argv)<2) :
        print("too few  arguments need 2 as minimum , image path, and model name")
        print("command should look like this predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3 ")
        return
    
    elif  chek_image_path(args.image)==False:
        print("images path is not provided ")
        print("command should look like this predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3 ")
        return
    
    elif chek_model_path(args.model) == False:
        print("Model path is not provided, just put the model name without any path ")
        print("command should look like this predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3 ")
        return
    
    else:
        top_k=args.top_k if args.top_k else 5
        top_k_values,top_k_indices= predict(args.image,args.model,top_k)
        top_k_values = tf.nn.top_k(list(tf.cast(top_k_values, tf.float32)), k=top_k)
      

        if args.category_names != None:
            category_names =red_json_file(args.category_names,top_k_indices )
            print("predictions : ")
            print( np.array(top_k_values)[0])
            print("names are : ")
            print(category_names)
        else:
            print("predictions : ")
            print( np.array(top_k_values)[0])
            print("names are : ")
            print(top_k_indices)
        


# Call to main function to run the program
if __name__ == "__main__":
    main()