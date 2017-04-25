#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  if (estimations.size() != ground_truth.size() || estimations.size()<=0){
    std::cout << "Error in calculating RMSE " << std::endl;
    return rmse;
  }
  for (int i=0; i<estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  rmse = (rmse/estimations.size()).array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float c1 = px*px + py*py;
  if(fabs(c1) < 0.0001) {
    Hj.setZero();
    return Hj;
  }
  float c2 = sqrt(c1);
  float c3 = c1 * c2;
  float n1 = vx*py - vy*px;
  float n2 = vy*px - vx*py;
  Hj << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
        py*n1/c3, px*n2/c3, px/c2, py/c2;
  return Hj;
}

VectorXd Tools::CalculateStateMeasurement(const VectorXd& x_state) {
  VectorXd measurement(3);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float rho = sqrt(px*px + py*py);
  if(fabs(rho) < 0.0001 || fabs(px) < 0.0001) {
    measurement.setZero();
    return measurement;
  }
  float phi = atan2(py,px);
  float rho_dot = (px*vx+py*vy)/rho;
  measurement << rho, phi, rho_dot;
  return measurement;
}
