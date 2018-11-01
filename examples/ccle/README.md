# Example of HRTs applied to data from the Cancer Cell Line Encyclopedia

This is a case study applied to the predictive modeling approach in the following paper:

Barretina, J., Caponigro, G., Stransky, N., Venkatesan, K., Margolin, A. A., Kim, S., ... & Reddy, A. (2012). The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature, 483(7391), 603.

## Running

The code in `main.py` generates p-values for the top 10 features chosen by averaging elastic net coefficient weights in 10-fold cross-validation.

To run this code, you will need to download three data files:

- [mutation.txt](https://www.dropbox.com/s/7iy0ght31hxhn7d/mutation.txt?dl=0)

- [expression.txt](https://www.dropbox.com/s/bplwquwbc7zleck/expression.txt?dl=0)

- [response.csv](https://www.dropbox.com/s/78mp3ebnb4h6jsy/response.csv?dl=0)


