import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QCheckBox, QGridLayout, QGroupBox, QRadioButton,
    QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from utils.model_utils import load_scaler, load_metadata, load_model_ensemble_members
from utils.metrics import predict_with_uncertainty, convert_to_original_scale

MODEL_NAME = 'std5_norm'
BASE_OUTPUT_DIR = './trained_models'
MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, MODEL_NAME)

ITER_PARAMS = {
    'IP': 15.0,        # 8-16 MA
    'BT': 5.3,         # Fixed
    'GWF': 0.85,       # 0-1.2
    'DELTA': 0.48,     # Triangularity
    'PALPHA': 100.0,   # 80 or 100 MW
    'PNBI': 33.0,      # 0-100 MW
    'PECRH': 20.0,     # 0-100 MW
    'PICRH': 20.0,     # 0-100 MW
    'HIGHZ': 1,        # 0 or 1
    'COAT': 1,         # 0 or 1
    'RGEO': 6.2,       # Fixed - Major radius
    'AMIN': 2.0,       # Fixed - Minor radius
    'KAREA': 1.7,      # Fixed - Elongation
    'MEFF': 2.5,       # Fixed - Effective mass
}

SLIDER_SCALE = 100

class ITPA_BNN_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ITPA BNN H-factor Prediction Tool")
        self.resize(800, 500)
        
        self.model_artifacts = None
        self.load_model_artifacts()

        self.sliders = {}
        self.slider_labels = {}
        self.checkboxes = {}
        self.radio_buttons = {}
        self.button_groups = {}
        self.output_labels = {}
        
        self.init_ui()
        
        QTimer.singleShot(100, self.update_predictions)

    def load_model_artifacts(self):
        metadata = load_metadata(os.path.join(MODEL_DIR, 'metadata.json'))
        scaler_X = load_scaler(os.path.join(MODEL_DIR, 'scaler_X.joblib'))
        scaler_y = load_scaler(os.path.join(MODEL_DIR, 'scaler_y.joblib'))
        trained_models = load_model_ensemble_members(MODEL_DIR, metadata)
        self.model_artifacts = {
            'metadata': metadata, 
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y,
            'trained_models': trained_models, 
            'loaded': True
        }

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        input_panel = QWidget()
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)
        
        plasma_group = QGroupBox("Plasma Parameters")
        plasma_layout = QGridLayout()
        plasma_group.setLayout(plasma_layout)
        
        ip_label = QLabel("Plasma Current (MA)")
        ip_value_label = QLabel(f"{ITER_PARAMS['IP']:.1f}")
        ip_slider = QSlider(Qt.Horizontal)
        ip_slider.setRange(int(8 * SLIDER_SCALE), int(16 * SLIDER_SCALE))
        ip_slider.setValue(int(ITER_PARAMS['IP'] * SLIDER_SCALE))
        ip_slider.valueChanged.connect(self.update_predictions)
        plasma_layout.addWidget(ip_label, 0, 0)
        plasma_layout.addWidget(ip_slider, 0, 1)
        plasma_layout.addWidget(ip_value_label, 0, 2)
        self.sliders['IP'] = ip_slider
        self.slider_labels['IP'] = ip_value_label
        
        gwf_label = QLabel("Greenwald Fraction")
        gwf_value_label = QLabel(f"{ITER_PARAMS['GWF']:.2f}")
        gwf_slider = QSlider(Qt.Horizontal)
        gwf_slider.setRange(int(0 * SLIDER_SCALE), int(1.2 * SLIDER_SCALE))
        gwf_slider.setValue(int(ITER_PARAMS['GWF'] * SLIDER_SCALE))
        gwf_slider.valueChanged.connect(self.update_predictions)
        plasma_layout.addWidget(gwf_label, 1, 0)
        plasma_layout.addWidget(gwf_slider, 1, 1)
        plasma_layout.addWidget(gwf_value_label, 1, 2)
        self.sliders['GWF'] = gwf_slider
        self.slider_labels['GWF'] = gwf_value_label
        
        delta_label = QLabel("Triangularity")
        delta_value_label = QLabel(f"{ITER_PARAMS['DELTA']:.2f}")
        delta_slider = QSlider(Qt.Horizontal)
        delta_slider.setRange(int(0.2 * SLIDER_SCALE), int(0.8 * SLIDER_SCALE))
        delta_slider.setValue(int(ITER_PARAMS['DELTA'] * SLIDER_SCALE))
        delta_slider.valueChanged.connect(self.update_predictions)
        plasma_layout.addWidget(delta_label, 2, 0)
        plasma_layout.addWidget(delta_slider, 2, 1)
        plasma_layout.addWidget(delta_value_label, 2, 2)
        self.sliders['DELTA'] = delta_slider
        self.slider_labels['DELTA'] = delta_value_label

        input_layout.addWidget(plasma_group)
        
        heating_group = QGroupBox("Heating Power (MW)")
        heating_layout = QGridLayout()
        heating_group.setLayout(heating_layout)
        
        pnbi_label = QLabel("NBI Power")
        pnbi_value_label = QLabel(f"{ITER_PARAMS['PNBI']:.1f}")
        pnbi_slider = QSlider(Qt.Horizontal)
        pnbi_slider.setRange(0, int(100 * SLIDER_SCALE))
        pnbi_slider.setValue(int(ITER_PARAMS['PNBI'] * SLIDER_SCALE))
        pnbi_slider.valueChanged.connect(self.update_predictions)
        heating_layout.addWidget(pnbi_label, 0, 0)
        heating_layout.addWidget(pnbi_slider, 0, 1)
        heating_layout.addWidget(pnbi_value_label, 0, 2)
        self.sliders['PNBI'] = pnbi_slider
        self.slider_labels['PNBI'] = pnbi_value_label
        
        pecrh_label = QLabel("ECRH Power")
        pecrh_value_label = QLabel(f"{ITER_PARAMS['PECRH']:.1f}")
        pecrh_slider = QSlider(Qt.Horizontal)
        pecrh_slider.setRange(0, int(100 * SLIDER_SCALE))
        pecrh_slider.setValue(int(ITER_PARAMS['PECRH'] * SLIDER_SCALE))
        pecrh_slider.valueChanged.connect(self.update_predictions)
        heating_layout.addWidget(pecrh_label, 1, 0)
        heating_layout.addWidget(pecrh_slider, 1, 1)
        heating_layout.addWidget(pecrh_value_label, 1, 2)
        self.sliders['PECRH'] = pecrh_slider
        self.slider_labels['PECRH'] = pecrh_value_label
        
        picrh_label = QLabel("ICRH Power")
        picrh_value_label = QLabel(f"{ITER_PARAMS['PICRH']:.1f}")
        picrh_slider = QSlider(Qt.Horizontal)
        picrh_slider.setRange(0, int(100 * SLIDER_SCALE))
        picrh_slider.setValue(int(ITER_PARAMS['PICRH'] * SLIDER_SCALE))
        picrh_slider.valueChanged.connect(self.update_predictions)
        heating_layout.addWidget(picrh_label, 2, 0)
        heating_layout.addWidget(picrh_slider, 2, 1)
        heating_layout.addWidget(picrh_value_label, 2, 2)
        self.sliders['PICRH'] = picrh_slider
        self.slider_labels['PICRH'] = picrh_value_label
        
        input_layout.addWidget(heating_group)
        
        alpha_impurity_group = QGroupBox("Alpha Power & Impurities")
        alpha_impurity_layout = QGridLayout()
        alpha_impurity_group.setLayout(alpha_impurity_layout)
        
        palpha_label = QLabel("Alpha Heating (MW)")
        palpha_group = QButtonGroup(self)
        self.button_groups['PALPHA'] = palpha_group
        
        palpha_80 = QRadioButton("80")
        palpha_100 = QRadioButton("100")
        palpha_100.setChecked(True)
        
        palpha_group.addButton(palpha_80, 80)
        palpha_group.addButton(palpha_100, 100)
        palpha_group.buttonClicked.connect(self.update_predictions)
        
        alpha_impurity_layout.addWidget(palpha_label, 0, 0)
        alpha_impurity_layout.addWidget(palpha_80, 0, 1)
        alpha_impurity_layout.addWidget(palpha_100, 0, 2)
        
        highz_cb = QCheckBox("High-Z Impurities")
        highz_cb.setChecked(bool(ITER_PARAMS['HIGHZ']))
        highz_cb.stateChanged.connect(self.update_predictions)
        alpha_impurity_layout.addWidget(highz_cb, 1, 0, 1, 3)
        self.checkboxes['HIGHZ'] = highz_cb
        
        coat_cb = QCheckBox("Wall Coating")
        coat_cb.setChecked(bool(ITER_PARAMS['COAT']))
        coat_cb.stateChanged.connect(self.update_predictions)
        alpha_impurity_layout.addWidget(coat_cb, 2, 0, 1, 3)
        self.checkboxes['COAT'] = coat_cb
        
        input_layout.addWidget(alpha_impurity_group)
        
        output_panel = QWidget()
        output_layout = QVBoxLayout()
        output_panel.setLayout(output_layout)
        
        prediction_group = QGroupBox("Prediction Results")
        prediction_layout = QVBoxLayout()
        prediction_group.setLayout(prediction_layout)
        
        h_label = QLabel("H98(y,2): N/A")
        h_label.setFont(QFont("Arial", 14, QFont.Bold))
        prediction_layout.addWidget(h_label)
        self.output_labels['H98Y2'] = h_label
        
        h_std_label = QLabel("H98(y,2) Std Dev: N/A")
        prediction_layout.addWidget(h_std_label)
        self.output_labels['H98Y2_STD'] = h_std_label
        
        output_layout.addWidget(prediction_group)
        
        calc_group = QGroupBox("Calculated Parameters")
        calc_layout = QGridLayout()
        calc_group.setLayout(calc_layout)
        
        calc_params = [
            ('qCYL', "Safety Factor qCYL"),
            ('NEL', "Electron Density NEL (1e19 m⁻³)"),
            ('PLH', "L-H Threshold PLH (MW)"),
            ('PFIN', "PFIN = (PNBI + PALPHA) / PLH"),
            ('PRFN', "PRFN = (PECRH + PICRH) / PLH"),
            ('PTOT', "Total Heating Power (MW)")
        ]
        
        for i, (key, label_text) in enumerate(calc_params):
            label = QLabel(f"{label_text}: N/A")
            calc_layout.addWidget(label, i, 0)
            self.output_labels[key] = label
        
        output_layout.addWidget(calc_group)
        
        input_layout.addStretch(1)
        output_layout.addStretch(1)
        
        main_layout.addWidget(input_panel)
        main_layout.addWidget(output_panel)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    @Slot()
    def update_predictions(self):
        if not self.model_artifacts or not self.model_artifacts.get('loaded', False):
            self.show_error_message("Model not loaded")
            return
            
        params = self.get_parameters_from_ui()
        derived_params = self.calculate_parameters(params)
        
        self.update_slider_labels(params)
        self.update_output_labels(derived_params)
        
        self.predict_h_factor(params, derived_params)

    def get_parameters_from_ui(self):
        params = ITER_PARAMS.copy()
        
        for key, slider in self.sliders.items():
            params[key] = slider.value() / SLIDER_SCALE
            
        for key, checkbox in self.checkboxes.items():
            params[key] = 1 if checkbox.isChecked() else 0
            
        if 'PALPHA' in self.button_groups:
            params['PALPHA'] = self.button_groups['PALPHA'].checkedId()
            
        return params

    def calculate_parameters(self, params):
        derived = {}
        
        derived['NEL'] = self.calculate_nel(params['GWF'], params['IP'], params['AMIN'])
        derived['qCYL'] = self.calculate_qcyl(params['BT'], params['IP'], 
                                            params['AMIN'], params['RGEO'], 
                                            params['KAREA'])
        derived['PLH'] = self.calculate_plh(derived['NEL'], params['BT'], 
                                          params['AMIN'], params['RGEO'], 
                                          params['MEFF'])
        derived['PFIN'] = self.calculate_pfin(params['PNBI'], params['PALPHA'], 
                                            derived['PLH'])
        derived['PRFN'] = self.calculate_prfn(params['PECRH'], params['PICRH'], 
                                            derived['PLH'])
        derived['PTOT'] = params['PNBI'] + params['PECRH'] + params['PICRH']
        
        return derived

    def calculate_nel(self, gwf, ip, amin):
        return gwf * ip / (np.pi * amin**2) * 10

    def calculate_qcyl(self, bt, ip, amin, rgeo, karea):
        return 5 * (bt / ip) * (amin**2 / rgeo) * karea

    def calculate_plh(self, nel, bt, amin, rgeo, meff):
        n20 = nel * 0.1
        return 2.15 * (n20**0.782) * (bt**0.772) * (amin**0.975) * rgeo * (2/meff)

    def calculate_pfin(self, pnbi, palpha, plh):
        return (pnbi + palpha) / plh if plh > 0 else 0

    def calculate_prfn(self, pecrh, picrh, plh):
        return (pecrh + picrh) / plh if plh > 0 else 0

    def update_slider_labels(self, params):
        for key, label in self.slider_labels.items():
            if key in ['GWF', 'DELTA']:
                label.setText(f"{params[key]:.2f}")
            else:
                label.setText(f"{params[key]:.1f}")

    def update_output_labels(self, derived_params):
        for key, value in derived_params.items():
            if key in self.output_labels:
                label = self.output_labels[key]
                if key == 'qCYL':
                    label.setText(f"Safety Factor qCYL: {value:.3f}")
                elif key == 'NEL':
                    label.setText(f"Electron Density NEL (1e19 m⁻³): {value:.3f}")
                elif key == 'PLH':
                    label.setText(f"L-H Threshold PLH (MW): {value:.2f}")
                elif key == 'PFIN':
                    label.setText(f"PFIN = (PNBI + PALPHA) / PLH: {value:.3f}")
                elif key == 'PRFN':
                    label.setText(f"PRFN = (PECRH + PICRH) / PLH: {value:.3f}")
                elif key == 'PTOT':
                    label.setText(f"Total Heating Power (MW): {value:.1f}")

    def predict_h_factor(self, params, derived_params):
        model_input = {}
        model_input['qCYL'] = derived_params['qCYL']
        model_input['GWF'] = params['GWF']
        model_input['DELTA'] = params['DELTA']
        model_input['PFIN'] = derived_params['PFIN']
        model_input['PRFN'] = derived_params['PRFN']
        model_input['HIGHZ'] = params['HIGHZ']
        model_input['COAT'] = params['COAT']
        
        metadata = self.model_artifacts['metadata']
        model_input_cols = metadata['input_cols']
        exclude_indices = metadata.get('exclude_scaling_indices', [])
        
        input_vector = []
        for col in model_input_cols:
            input_vector.append(model_input[col])
                
        input_vector = np.array(input_vector).reshape(1, -1)
        
        scaler_X = self.model_artifacts['scaler_X']
        if exclude_indices and len(exclude_indices) > 0:
            num_features = input_vector.shape[1]
            positive_exclude_indices = [idx if idx >= 0 else num_features + idx for idx in exclude_indices]
            cols_to_scale_mask = np.ones(num_features, dtype=bool)
            cols_to_scale_mask[positive_exclude_indices] = False
            
            scaled_parts = scaler_X.transform(input_vector[:, cols_to_scale_mask])
            input_vector_scaled = np.zeros_like(input_vector, dtype=float)
            input_vector_scaled[:, cols_to_scale_mask] = scaled_parts
            input_vector_scaled[:, ~cols_to_scale_mask] = input_vector[:, ~cols_to_scale_mask]
        else:
            input_vector_scaled = scaler_X.transform(input_vector)
                
        scaler_y = self.model_artifacts['scaler_y']
        trained_models = self.model_artifacts['trained_models']
        
        mean_scaled, epistemic_unc, aleatoric_unc, total_unc = predict_with_uncertainty(
            trained_models, input_vector_scaled
        )
        
        h_mean_arr, _, _, h_std_arr = convert_to_original_scale(
            mean_scaled, (epistemic_unc, aleatoric_unc, total_unc), scaler_y
        )
        
        h_mean = h_mean_arr.item()
        h_std = h_std_arr.item()
        
        self.output_labels['H98Y2'].setText(f"H98(y,2): {h_mean:.3f}")
        self.output_labels['H98Y2_STD'].setText(f"H98(y,2) Std Dev: {h_std:.3f}")

    def show_error_message(self, message):
        for key in self.output_labels:
            self.output_labels[key].setText(f"{key}: Error - {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ITPA_BNN_GUI()
    window.show()
    sys.exit(app.exec())