/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <utility>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    double std_x = std[0];
    double std_y = std[1];
    double std_yaw = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_psi(theta, std_yaw);

    weights.resize(static_cast<unsigned long>(num_particles));
    particles.resize(static_cast<unsigned long>(num_particles));
    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_psi(gen);
        particle.weight = 1.0 / num_particles;
        particles[i] = particle;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    if (yaw_rate == 0) {
        return;
    }

    double yaw_dt = yaw_rate * delta_t;
    double v_yaw = velocity / yaw_rate;

    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (auto &particle : particles) {
        double n_theta = particle.theta + yaw_dt;

        // predicting the next X
        particle.x += v_yaw * (sin(n_theta) - sin(particle.theta)) + dist_x(gen);

        // predicting the next Y
        particle.y += v_yaw * (cos(particle.theta) - cos(n_theta)) + dist_y(gen);

        // predicting the next Yaw
        particle.theta = n_theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (auto &obs: observations) {
        double min = 1e+100, d;
        for (auto const &landmark: predicted) {
            if (min > (d = dist(landmark.x, landmark.y, obs.x, obs.y))) {
                min = d;
                obs.id = landmark.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    double std_x = std_landmark[0], std_y = std_landmark[1];

    double d_sq_x = 2 * std_x * std_x;
    double d_sq_y = 2 * std_y * std_y;

    for (auto &particle: particles) {
        std::vector<LandmarkObs> landmarks_in_range;

        double sin_theta = sin(particle.theta);
        double cos_theta = cos(particle.theta);

        for (auto &landmark: map_landmarks.landmark_list) {
            if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range) {
                LandmarkObs landmark_obs = {int(landmarks_in_range.size()), double(landmark.x_f), double(landmark.y_f)};
                landmarks_in_range.push_back(landmark_obs);
            }
        }

        std::vector<LandmarkObs> transformed_obs;
        for (auto &landmark: observations) {

            double transformed_x = landmark.x * cos_theta - landmark.y * sin_theta + particle.x;
            double transformed_y = landmark.x * sin_theta + landmark.y * cos_theta + particle.y;
            LandmarkObs obs = {
                    landmark.id,
                    transformed_x,
                    transformed_y
            };
            transformed_obs.push_back(obs);
        }

        dataAssociation(landmarks_in_range, transformed_obs);

        double sum_sqr_x_diff = 0.0, sum_sqr_y_diff = 0.0;
        for (auto const &obs: transformed_obs) {
            double x_diff = obs.x - landmarks_in_range[obs.id].x;
            double y_diff = obs.y - landmarks_in_range[obs.id].y;
            sum_sqr_x_diff += pow(x_diff, 2);
            sum_sqr_y_diff += pow(y_diff, 2);
        }

        particle.weight = exp(-sum_sqr_x_diff / d_sq_x - sum_sqr_y_diff / d_sq_y);
    }

    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::discrete_distribution<int> dist_particle(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;
    resampled_particles.reserve(static_cast<unsigned long>(num_particles));

    for (int i = 0; i < num_particles; i++) {
        resampled_particles.push_back(particles[dist_particle(gen)]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = std::move(associations);
    particle.sense_x = std::move(sense_x);
    particle.sense_y = std::move(sense_y);

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
