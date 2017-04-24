#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.P_ = MatrixXd(4,4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    //cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_ << ro*cos(phi), ro*sin(phi), ro_dot*cos(phi), ro_dot*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      float x = measurement_pack.raw_measurements_[0];
      float y = measurement_pack.raw_measurements_[1];
      ekf_.x_ << x, y, 0, 0;
    }

    cout << "Init: "<< ekf_.x_(0) << " " << ekf_.x_(1)<< " " << ekf_.x_(2)<< " " << ekf_.x_(3) << endl;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  float noise_ax = 9.0, noise_ay = 9.0;
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000;
  float dt2 = dt*dt;
  previous_timestamp_ = measurement_pack.timestamp_;
  MatrixXd a(2,2), G(4,2);
  a << noise_ax, 0,
       0, noise_ay;
  G << dt2/2, 0,
       0, dt2/2,
       dt, 0,
       0, dt;
  ekf_.Q_ = G*a*G.transpose();

  float dt_2 = dt   * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
             0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
             dt_3 / 2 * noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3 / 2 * noise_ay, 0, dt_2*noise_ay;


  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  ekf_.Predict();
  cout << "Pred: "<< ekf_.x_(0) << " " << ekf_.x_(1)<< " " << ekf_.x_(2)<< " " << ekf_.x_(3) << endl;
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    Tools tools;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    if(!ekf_.H_.isZero()){
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
      cout << "RDAR: "<< ekf_.x_(0) << " " << ekf_.x_(1)<< " " << ekf_.x_(2)<< " " << ekf_.x_(3) << endl;
    } else {
      cout << "Skip: "<< ekf_.x_(0) << " " << ekf_.x_(1)<< " " << ekf_.x_(2)<< " " << ekf_.x_(3) << endl;
    }
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = MatrixXd(2,4);
    ekf_.H_ << 1, 0, 0, 0,
               0, 1, 0, 0;
    ekf_.Update(measurement_pack.raw_measurements_);
    cout << "LDAR: "<< ekf_.x_(0) << " " << ekf_.x_(1)<< " " << ekf_.x_(2)<< " " << ekf_.x_(3) << endl;
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
