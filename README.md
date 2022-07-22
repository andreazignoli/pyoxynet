<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="pics/logo_BW.png" alt="Logo" width="120" height="120">
  </a>
  <h3 align="center">The Oxynet Python package repository</h3>
  <p align="center">
    <a> :earth_africa: </a>
    <br />
    <a> We require world-wide coordinated actions for timely diagnostics.</a>
    <br />
        <az> :hospital: </a>
    <br />
    <a> Oxynet research project contributes with more equitable health care services.</a>
    <br />
        <a> :computer: </a>
    <br />
    <a> Oxynet is a set of tools for automatic interpretation of cardiopulmonary exercising tests data.</a>
    <br />
    <br />
    <a href="http://oxynet.net"><strong>Visit the website »</strong></a>
    <br />
    <a href="https://www.overleaf.com/read/zgsfxmvcbhkz">Overleaf</a>
    ·
    <a href="https://oxynetresearch.promfacility.eu">Web app</a>
    ·
    <a href="https://pypi.org/project/pyoxynet/">Pypi</a>
    ·
    <a href="https://pyoxynet.readthedocs.io/en/latest/index.html">Docs</a>
  </p>
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

## The *Pyoxynet* package

*Pyoxynet* is a collection of the algorithms developed within the *Oxynet* project. The core algorithms (i.e.models) are mainly deep neural networks, made in the attempt to process cardiopulmonary exercise test data (CPET). 

All the models have been trained and tested with [Tensorflow](https://www.tensorflow.org/), but they are included in *Pyoxynet* only in their [TFLite](https://www.tensorflow.org/lite) inference version. TFLite has been intentionally adopted to keep the package light and to promote the use of *Oxynet* related technologies. 

To date, mainly two type of models are implemented: 

* The *inference* model: it takes some CPET data as input and it provides an estimation of the exercise intensity domains 
* The *generator* model: it generates new synthetic CPET data

You can read more about the rationale and the technology behind the Oxynet project at the following links: 

* [Review](https://link.springer.com/article/10.1007%2Fs11332-019-00557-x) paper on the AI technologies applied to exercise cardiopulmonary and metabolic data processing
* [Research](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229466) implementing an LSTM neural networks to the estimation of VO2 during cycling exercise (regressor)
* [Research](https://www.tandfonline.com/doi/abs/10.1080/17461391.2019.1587523?journalCode=tejs20) implementing an LSTM neural networks to the estimation of the intensity domain during incremental exercise (classifier)
* [Research](https://www.tandfonline.com/doi/abs/10.1080/17461391.2020.1866081?journalCode=tejs20) implementing a crowd sourcing and CNN inference to the problem of determining the intensity domain during incremental exercise (classifier)
* [Research](https://www.overleaf.com/read/fcmwscvyhtfq) generating synthetic CPET data with conditional GANs
* [Blog article](https://www.linkedin.com/pulse/oxynet-collective-intelligence-approach-test-andrea-zignoli/) about the Oxynet project
* [Blog article](https://andreazignoli.github.io/blog-post-5/) about the problem of adopting AI models in the interpretation of CPET data

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

There is no need to clone this repo if you don't want to. You can just install the public Python package or call the public Flask-Pyoxynet service from a web app, a terminal on a server or from your browser. 

### Pip install the package

☝️ This package was developed under **Python 3.8**, so it might not work properly under older versions.   

This is the best solution for those Python users who would like to have *Oxynet* algorithms always on the tip of their fingers. Assuming you have pip installed on your machine, begin with: 

```sh
pip install pyoxynet
```

Or, alternatively, 

```sh
pip install git+https://github.com/andreazignoli/pyoxynet.git#subdirectory=pyoxynet
```

Packages that require addition extra url cannot be installed via *setuptools*, which lately allows and suggests to use *pip* when possible. To workaround this problem, TFLite is automatically installed with the following command the first time *pyoxynet* is imported:

```sh
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Currently, The TFLite installation process is completed with a line command inside Python, which is not the best solution (I know...). 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Data required for the *inference* include oxygen uptake (VO2), exhaled CO2 (VCO2), minute ventilation (VE), end tidal O2 (PetO2) and CO2(PetCO2), and ventilatory equivalents (VEVO2 and VEVCO2):

| VO2 | VCO2 | VE | PetO2 | PetCO2 | VEVO2 | VEVCO2 |
|-----|------|----|-------|--------|-------|--------|
|     |      |    |       |        |       |        |
|     |      |    |       |        |       |        |
|     |      |    |       |        |       |        |

*Oxynet* inference models work on data over-sampled on a sec-by-sec basis. When dealing with breath-by-breath data, linear interpolation at 1 second is appropriate. When dealing with averaged 5-by-5 second data or 10-by-10 second data, cubic interpolation is more appropriate. *Pyoxynet* however, can implement a number of interpolation algorithm to process raw data as well as data already processed. 

In case there is no access to VCO2 data, a different solution has been implemented (although validation has not been published) considering the following inputs: 

| VO2 | VE | PetO2 | Rf | VEVO2 |
|-----|----|-------|----|-------|
|     |    |       |    |       |
|     |    |       |    |       |
|     |    |       |    |       |

If you want to see how *Pyoxynet* can work on sample data:

```python
import pyoxynet

# Load the TFL model
tfl_model = pyoxynet.load_tf_model()

# Make inference on a random input
test_tfl_model(tfl_model)

# Plot the inference on a test dataset
pyoxynet.test_pyoxynet()
```

_For more examples, please refer to the package [Documentation](https://pyoxynet.readthedocs.io/en/latest/index.html)_

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GENERATION -->
## Generation

*Pyoxynet* also implements a Conditional Generative Adversarial Network, which has been trained to generate deceptive CPET data. As per the *inference* model, the *generator* is saved in a TFLite model file. Calling the related function and obtain a fake CPET data sample can be done as follows: 

```python
from pyoxynet import *
# Call the generator
generator = load_tf_generator()
# Generate a Pandas df with fake CPET data inside
df = generate_CPET(generator, plot=True)
# Call Oxynet for inference on fake data
test_pyoxynet(input_df=df)
```

In the gif below, different epochs/steps of the training process are presented for the Conditional Adversarial Neural Network available in *Pyoxynet*. 

![plot](./pics/dcgan_200.gif)

<p align="right">(<a href="#top">back to top</a>)</p>

## Flask-Pyoxynet inference app

A [Flask](https://flask.palletsprojects.com/en/2.0.x/) inference/generation app called Flask-Pyoxynet has been deployed on a [Amazon Lightsail](https://aws.amazon.com/getting-started/hands-on/serve-a-flask-app/) private server. Currently, flask-pyoxynet runs on Lightsail containers service. 

It is possible therefore to call Flask-Pyoxynet from a terminal, and provide data in json format. If your input data has only 7 variables, then the classic Oxynet configuration can be used by replacing X with 7 (see command below), otherwise if the input variables are only 5, you can replace X with 5:

```sh
curl -X POST https://flask-service.ci6m7bo8luvmq.eu-central-1.cs.amazonlightsail.com/read_json?n_inputs=X -d @test_data.json
```

It is possible to check the required keys of the json dictionary in *app/test_data/test_data.json*. Alternatively, it is possible to directly check the *generated* example at this [address](https://flask-service.ci6m7bo8luvmq.eu-central-1.cs.amazonlightsail.com/CPET_plot). It is also possible to directly retrieve *generated* data in *json* format at this [address](https://flask-service.ci6m7bo8luvmq.eu-central-1.cs.amazonlightsail.com/CPET_generation).

```sh
https://flask-service.ci6m7bo8luvmq.eu-central-1.cs.amazonlightsail.com
```

The app can also be used as test to check how realistic the fake examples look like at the [app main page](https://flask-service.ci6m7bo8luvmq.eu-central-1.cs.amazonlightsail.com/).  

<!-- ROADMAP -->
## Roadmap

- [x] Create web app for inference
- [x] Create web app for data crowd-sourcing
- [x] Create website
- [x] Create Python package
    - [x] Implement inference
    - [x] Implement generation
- [x] Develop Flask-app
- [x] Run Docker on AWS
- [x] Develop GUI for generation
    - [ ] Improve GUI
    - [ ] ----

See the [open issues](https://github.com/andreazignoli/pyoxynet/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing to the Oxynet project

There are challenges that transcend both national and continental boundaries and providing people with universal access to good quality health care is one of them. Emerging technologies in the field of AI and the availability of vast amounts of data can offer big opportunities to stimulate innovation and develop solutions.

*Oxynet* wants to become a tool for a quick and encompassing diagnosis of medical conditions with cardiopulmonary exercise tests (CPET) and promote accurate and timely clinical decisions, ultimately reducing the costs associated with current evaluation errors and delays.

The main building blocks of Oxynet are: 

* A network of experts in the field of CPET
* A large crowd sourced data set
* An AI algorithm able to approximate human cognition in the analysis of CPET 

We are interested in creating more research opportunities with other Universities and Departments, hospitals and clinics, medical doctors and physiologists (also operating in intensive care units), companies involved in the development (including patenting and validation) and in the commercialization of medical devices (e.g. metabolic carts and medical software). 

We want to bring together key actors from across sectors to jointly implement our R&D road map and: support the research activities financially (including scholarships for research fellows or publication fees for open access journals), provide intellectual contribution for scientific publications or grant application, share data for testing/developing new algorithms, develop web-based applications (e.g. crowd sourcing applications, automatic interpretation of new data, websites for communicating the outcomes of the project), conduct market and patent analyses, and validate the algorithms for clinical settings.

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give Pyoxynet a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Andrea Zignoli - [@andrea_zignoli](https://twitter.com/andrea_zignoli) - andrea.zignoli@unitn.it

Repository project link: [Pyoxynet](https://github.com/andreazignoli/pyoxynet)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The following resources are valuable for both the *Pyoxynet* and *Oxynet* projects:

* [TFLite inference](https://www.tensorflow.org/lite/guide/inference)
* [Amazon Lightsail](https://aws.amazon.com/getting-started/hands-on/serve-a-flask-app/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [Uniplot Python library](https://github.com/olavolav/uniplot)
* [Machine Learning Mastery cGAN](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/andreazignoli/pyoxynet.svg?style=for-the-badge
[contributors-url]: https://github.com/andreazignoli/pyoxynet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/andreazignoli/pyoxynet.svg?style=for-the-badge
[forks-url]: https://github.com/andreazignoli/pyoxynet/network/members
[stars-shield]: https://img.shields.io/github/stars/andreazignoli/pyoxynet.svg?style=for-the-badge
[stars-url]: https://github.com/andreazignoli/pyoxynet/stargazers
[issues-shield]: https://img.shields.io/github/issues/andreazignoli/pyoxynet.svg?style=for-the-badge
[issues-url]: https://github.com/andreazignoli/pyoxynet/issues
[license-shield]: https://img.shields.io/github/license/andreazignoli/pyoxynet.svg?style=for-the-badge
[license-url]: https://github.com/andreazignoli/pyoxynet/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/andrea-zignoli-8080a438/
[product-screenshot]: images/screenshot.png

# Disclaimer

All content found on this website, including: text, images, tables, or other formats are created for informational purposes only. The information provided by this software is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something has been provided by this software.

<p align="right">(<a href="#top">back to top</a>)</p>