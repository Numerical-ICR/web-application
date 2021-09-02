# Numerical-ICR

Numerical-ICR is a numerical intelligent character recognition tool.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of the required libairies

```bash
$ pip install -r requirements.txt
```


## Setup
1. Open the `.sample.env` file from and add a random string for "FLASK_SECRET_KEY", you can use an online tool to randomlly generate this.
2. Execute the application using the instruction below.
3. Train the model first, this may take 5-10 minutes.

## Execution

```bash
flask run
```

## Usage
On the homepage of the application you can upload images, these images will be processed and the prediction will be generated on the results page. If any results are wrong then you have the ability to correct it or delete the row if it's a falsey detection character. Additionally, the results can be saved to as a txt file or the data can be saved for future training later.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)