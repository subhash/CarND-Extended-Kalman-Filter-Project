#include "kalman_filter.h"
#include "tools.h"
#include "math.h"
#include <iostream>


using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;


KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::UpdateState(const VectorXd &y) {
  MatrixXd PHt = P_*H_.transpose();
  MatrixXd S = H_*PHt + R_;
  MatrixXd K = PHt*S.inverse();
  const long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  x_ = x_ + K*y;
  P_ = (I - K*H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;

  UpdateState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;
  VectorXd hx = tools.CalculateStateMeasurement(x_);
  if (hx.isZero()){
    std::cout << "Skipping measurement conversion to avoid error" << x_ << std::endl;
    return;
  }
  VectorXd y = z - hx;
  y[1] = atan2(sin(y[1]), cos(y[1]));

  UpdateState(y);
}
