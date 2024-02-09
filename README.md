# Multifidelity continual learning

This repository contains the code for 

                 Howard, Amanda, Yucheng Fu, and Panos Stinis. "A multifidelity approach to continual learning for physical systems." arXiv preprint arXiv:2304.03894 (2023).


DISCLAIMER:
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830


# Data 

##The data for Section 4.2 is from
> *Wang, Z., Hong, T., Li, H. and Piette, M.A., 2021. Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models. Advances in Applied Energy, p.100025.*
https://doi.org/10.1016/j.adapen.2021.100025
[Zhe Wang](mailto:zwang5@lbl.gov ) or [Tianzhen Hong](mailto:thong@lbl.gov)

Access to original data: 
git clone https://github.com/LBNL-ETA/City-Scale-Electricity-Use-Prediction

@article{wang2021predicting,
  title={Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models},
  author={Wang, Zhe and Hong, Tianzhen and Li, Han and Piette, Mary Ann},
  journal={Advances in Applied Energy},
  pages={100025},
  year={2021},
  publisher={Elsevier}
}


# MAS implementation

The MAS implementation is adapted from 
https://github.com/ariseff/overcoming-catastrophic

@misc{ariseff,
  author = {Seff, Ari},
  title = {overcoming-catastrophic},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ariseff/overcoming-catastrophic}},
  commit = {cab6d86}
}

# Jax neural network

The Jax neural network implementation is adapted from 
https://github.com/PredictiveIntelligenceLab/ImprovedDeepONets

@misc{sifanw094,
  author = {Wang, Sifan},
  title = {ImprovedDeepONets},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PredictiveIntelligenceLab/ImprovedDeepONets
}},
  commit = {f948cf3}
}

# Citation
@article{howard2023multifidelity,
  title={A multifidelity approach to continual learning for physical systems},
  author={Howard, Amanda and Fu, Yucheng and Stinis, Panos},
  journal={arXiv preprint arXiv:2304.03894},
  year={2023}
}

