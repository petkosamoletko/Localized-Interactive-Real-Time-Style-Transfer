# Built-in library imports
import sys
import re
import functools
from functools import partial
import locale

# External library imports
import numpy as np
import torch
import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QRadioButton, QButtonGroup, QScrollArea, QSizePolicy,
    QGridLayout, QStyleFactory, QSlider, QLineEdit, QDoubleSpinBox, QComboBox,
    QCheckBox, QMessageBox, QGroupBox)
from PySide6.QtGui import QPixmap, QPainter, QPen, QMouseEvent, QImage, QFont
from PySide6.QtCore import Qt, QPoint
from segment_anything import SamPredictor, sam_model_registry

# Local modules
from network import modified_AdaIN_network
from utils import (change_hsv, content_loader, style_loader, mask_loader, 
                        stylized_output_converter, dict_keys_update)

class InteractiveStyleTransfer(QWidget):
    def __init__(self):
        super().__init__()
        
        # UI - Related
        self.setWindowTitle("Interactive Style Transfer Interface")
        self.content_image_display = QLabel()
        self.statusLabel = QLabel("Normal mode")  
        
        # Initilization
        self.prompt_points_coordinates = []
        self.click_types = []
        self.segmentation_pane_masks = []
        self.user_selected_finalized_masks = {}
        self.style_uploaded_images = {}
        self.redo_buttons = {}
        self.hexColorInputs = {} 
        self.maskColorToggleButtons = {}  
        self.segmentation_pane_selected_mask_id = 0 
        self.allowed_to_annotate = False
        self.redo_mode = False
        self.preserve_content_color = True
        self.toggle_button_before_vs_after = False
        self.content_image_path = None
        self.redo_mask_id = None  
        self.masks_with_uploaded_styles = set()

        #  GUI components initializaiton 
        self.initUI()
        self.content_pixmap = None
        self.maskWidget = QWidget()
        self.layout.addWidget(self.maskWidget)
        self.toggleButton = None
        
        # SAM Initiliazations 
        self.init_predictor()
        self.init_style_transfer_network()

        # Interactive Canvas Title
        self.canvasTitleLabel = QLabel("Interactive Image Canvas")
        self.canvasTitleLabel.setFont(QFont('Arial', 14, QFont.Bold))
        self.canvasTitleLabel.setAlignment(Qt.AlignLeft)
        self.canvasLayout = QVBoxLayout()
        self.canvasLayout.addWidget(self.canvasTitleLabel)
        self.canvasLayout.addWidget(self.promptable_content)
        
        self.interactiveCanvas.addLayout(self.canvasLayout)

    def initUI(self):
        # Application Style
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(QApplication.style().standardPalette())

        # Primary widgets 
        self.layout = QVBoxLayout()
        self.interactiveCanvas = QHBoxLayout()
        self.promptable_content = QLabel()
        self.interactiveCanvas.addWidget(self.promptable_content)
        self.buttonLayout = QVBoxLayout()
        self.scrollAbility = QScrollArea()
        self.scrollAbility.setWidgetResizable(True)
        self.scrollAbility.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Initially "on start" displayed buttons
        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.clicked.connect(self.upload_image)

        self.analyzeImageButton = QPushButton("Analyze Image")
        self.analyzeImageButton.clicked.connect(self.analyze_image)
        self.analyzeImageButton.setEnabled(False)

        self.undoButton = QPushButton("Undo")
        self.undoButton.clicked.connect(self.undo_annotation)
        self.undoButton.setEnabled(False)

        self.clearSelectionButton = QPushButton("Clear selection")
        self.clearSelectionButton.clicked.connect(self.clear_annotations)
        self.clearSelectionButton.setEnabled(False)

        self.segmentButton = QPushButton("Segment")
        self.segmentButton.clicked.connect(self.segment_image)
        self.segmentButton.setEnabled(False)

        self.finishedSegmentButton = QPushButton("I have finished segmenting")
        self.finishedSegmentButton.clicked.connect(self.style_transfer_pane_initiliazation)
        self.finishedSegmentButton.setEnabled(False)

        self.resetButton = QPushButton("Start all over again")
        self.resetButton.clicked.connect(self.reset_all)
        self.resetButton.setEnabled(False)

        self.exitButton = QPushButton("Exit")
        self.exitButton.clicked.connect(self.close_application)
        self.exitButton.setEnabled(True)

        # Grouped Layout of buttons
        # Upload group
        upload_group_layout = QVBoxLayout()
        upload_group_layout.addWidget(self.uploadButton)
        upload_group = QGroupBox("Upload")
        upload_group.setFont(QFont('Arial', 13, QFont.Bold))
        upload_group.setLayout(upload_group_layout)

        # SAM group
        sam_group_layout = QVBoxLayout()
        sam_group_layout.addWidget(self.analyzeImageButton)
        undo_clear_layout = QHBoxLayout()
        undo_clear_layout.addWidget(self.undoButton)
        undo_clear_layout.addWidget(self.clearSelectionButton)
        sam_group_layout.addLayout(undo_clear_layout)
        sam_group_layout.addWidget(self.segmentButton)
        finished_segment_layout = QVBoxLayout()
        finished_segment_layout.addWidget(self.finishedSegmentButton)
        sam_group_layout.addLayout(finished_segment_layout)
        sam_group = QGroupBox("SAM")
        sam_group.setFont(QFont('Arial', 13, QFont.Bold))
        sam_group.setLayout(sam_group_layout)

        # New project & Exit group
        exit_group_layout = QVBoxLayout()
        exit_group_layout.addWidget(self.resetButton)
        exit_group_layout.addWidget(self.exitButton)
        exit_group = QGroupBox("Start all over again or Exit")
        exit_group.setFont(QFont('Arial', 13, QFont.Bold))
        exit_group.setLayout(exit_group_layout)

        # Group Button Sizing Properties
        upload_group.setMaximumWidth(600)
        upload_group.setMaximumHeight(100)
        upload_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sam_group.setMaximumWidth(600)
        sam_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        exit_group.setMaximumWidth(600)
        exit_group.setMaximumHeight(150)
        exit_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Orgnization of elements 
        button_groups_layout = QVBoxLayout()
        button_groups_layout.addWidget(upload_group)
        button_groups_layout.addWidget(sam_group)
        button_groups_layout.addWidget(exit_group)

        image_and_buttons_layout = QHBoxLayout()
        image_and_buttons_layout.addLayout(self.interactiveCanvas)
        image_and_buttons_layout.addLayout(button_groups_layout)
        self.layout.addLayout(image_and_buttons_layout)

        content = QWidget()
        content.setLayout(self.layout)
        self.scrollAbility.setWidget(content)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scrollAbility)
        self.setLayout(main_layout)

    def close_application(self):
        '''
        Exit button functionaltiy
        '''
        self.close()

    def init_predictor(self):
        '''
        Segment Anything (SAM) set-up
        '''
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "default"
        # If apple metal is available utalize it.
        # Cuda utalization has been excluded due to the high chance
        # of running out of memory if utalized on a less capable GPU
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"SAM is running on {device}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def upload_image(self):
        '''
        Upload button functionality
        '''
        self.content_image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if self.content_image_path:
            self.content_pixmap = QPixmap(self.content_image_path)
            content_image = QImage(self.content_image_path)  
            self.promptable_content.setAlignment(Qt.AlignCenter) 
            # Scale down the content image if it is beyond either 1920 / 1080 in some if its dimensions
            if content_image.width() > 1920 or content_image.height() > 1080:
                self.content_pixmap = self.content_pixmap.scaled(1600, 900, Qt.KeepAspectRatio) 
                # If the image is downsized, let it be saved in the same location as the original content image
                downsized_content_image_path = self.content_image_path.rsplit('.', 1)[0] + "_downsized." + self.content_image_path.rsplit('.', 1)[1]
                self.content_pixmap.save(downsized_content_image_path)
                # Replace content_image_path with the now downsized version
                self.content_image_path = downsized_content_image_path 

            # Have one display of the image and one of the promptable overlay
            self.promptable_content.setPixmap(self.content_pixmap)
            self.content_image_display.setPixmap(self.content_pixmap)
            
           # Alter the availability of buttons 
            self.analyzeImageButton.setEnabled(True)
            self.undoButton.setEnabled(False)
            self.clearSelectionButton.setEnabled(False)
            self.segmentButton.setEnabled(False)
            self.finishedSegmentButton.setEnabled(False)
            self.resetButton.setEnabled(False)
            
    def analyze_image(self):
        '''
        Analyze button functionality
        '''
        if self.content_image_path:
            content_img_raw = cv2.imread(self.content_image_path)
            opencv_image = cv2.cvtColor(content_img_raw, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(opencv_image)
        
        # Alter the availability of buttons
        self.allowed_to_annotate = True
        self.undoButton.setEnabled(True)
        self.clearSelectionButton.setEnabled(True)
        self.segmentButton.setEnabled(True)
        self.finishedSegmentButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        self.analyzeImageButton.setEnabled(False)

    def mousePressEvent(self, click: QMouseEvent):
        '''
        Responsible for annotations (prompts) in the Interactive Display
        '''
        if not self.allowed_to_annotate:
            return

        #  Check button click type left/right
        if self.content_image_path and click.button() in {Qt.LeftButton, Qt.RightButton}:
            # Adjusting the scaling between the orinigal pixmap and local positions
            promptable_width = self.promptable_content.width()
            promptable_height = self.promptable_content.height()
            content_width = self.content_pixmap.width()
            content_height = self.content_pixmap.height()

            scale_x = content_width / self.promptable_content.pixmap().width()
            scale_y = content_height / self.promptable_content.pixmap().height()

            pos = self.promptable_content.mapFromGlobal(click.globalPos())
            pos.setX((pos.x() - promptable_width//2 + content_width//2) * scale_x)
            pos.setY((pos.y() - promptable_height//2 + content_height//2) * scale_y)
            
            # Determine click type
            click_type = 1 if click.button() == Qt.LeftButton else 0
            # Check if the adjusted position is within the bounds of the original image pixmap
            x_pos, y_pos = int(pos.x()), int(pos.y())
            if 0 <= x_pos <= content_width and 0 <= y_pos <= content_height:
                # Store click type, alongside the coordinates of point 
                self.prompt_points_coordinates.append([x_pos, y_pos])  
                self.click_types.append(click_type)
                self.annotating_content_image_display()

    def annotating_content_image_display(self):
        '''
        Pressenting the annotation (prompting) in the Intractive Display
        associated with the segmentation process
        '''
        pixmap_copy = QPixmap(self.content_pixmap)
        for type_click, point in enumerate(self.prompt_points_coordinates):
            if self.click_types[type_click] == 1:
                color = Qt.red
            else:
                color = Qt.blue
            painter = QPainter(pixmap_copy)
            painter.setPen(QPen(color, 10))
            painter.drawPoint(QPoint(*point))
            painter.end()
        
        # Scale pixmap accordingly to the content image in the Intractive Display
        width = self.promptable_content.width()
        height = self.promptable_content.height()
        scaled_pixmap = pixmap_copy.scaled(width, height, Qt.KeepAspectRatio)

        self.promptable_content.setPixmap(scaled_pixmap)

    def undo_annotation(self):
        '''
        Undo annotation(prompting) button functionality
        '''
        if self.prompt_points_coordinates:
            self.prompt_points_coordinates.pop()
            self.click_types.pop()
            self.annotating_content_image_display()

    def clear_annotations(self):
        '''
        Clear annotations button functionality 
        '''
        # Refresh the lists for click types and coordinates
        self.prompt_points_coordinates = []
        self.click_types = []

        # Scale pixmap accordingly to the content image in the Intractive Display
        width = self.promptable_content.width()
        height = self.promptable_content.height()
        scaled_pixmap = self.content_pixmap.scaled(width, height, Qt.KeepAspectRatio)

        self.promptable_content.setPixmap(scaled_pixmap)

    def segment_image(self):
        '''
        Segment button functionality, leveraging SAM and the annotaitons created
        to generate binary masks
        '''
        # Error message if no prompts have been made 
        if not self.prompt_points_coordinates: 
            QMessageBox.warning(self, "No Prompts Made", 
                                "No annotations have been placed. Please place at least a single annotation before segmenting.")
            return 
        
        self._remove_segmentation_pane()
        # SAM functionality of plotting masks
        if self.content_image_path and self.prompt_points_coordinates and self.click_types:
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(self.prompt_points_coordinates),
                point_labels=np.array(self.click_types),
                multimask_output=True,
            )
            # Store & Display the generated binary masks 
            self.segmentation_pane_masks = masks 
            self.segmentation_pane_layout(masks)

    def segmentation_pane_layout(self, masks):
        '''
        Plotting layout the masks in the Segmentation (Mask Creation) Pane
        '''
        # Load binary masks generated by SAM 
        self.masks = masks
        self.maskWidgets = []

        self.maskSegmentationLayout = QGridLayout()
        self.maskButtonConfirmDiscard = QButtonGroup()
        self.maskButtonConfirmDiscard.setExclusive(True)  

        # Create and set up the title promptable_content for masks
        self.titleLabelSegmentation = QLabel("Segmentation (Mask Creation)")
        self.titleLabelSegmentation.setFont(QFont('Arial', 14, QFont.Bold))
        self.titleLabelSegmentation.setAlignment(Qt.AlignLeft)
        self.maskSegmentationLayout.addWidget(self.titleLabelSegmentation, 0, 0) 

        # Color for displaying masks
        np.random.seed(25)
        color = np.random.randint(0, 256, 3).tolist()

        for idx, mask in enumerate(masks):
            mask_widget = self._mask_plotting_segmentation_pane(mask, color)
            self.maskButtonConfirmDiscard.addButton(mask_widget["radio"], idx)
            self.maskWidgets.append(mask_widget["widget"])
            self.maskSegmentationLayout.addWidget(mask_widget["widget"], 1, idx) 

        self.confirm_and_discard_buttons(len(masks))
        self._clear_mask_space_segmentation_pane()


    def _mask_plotting_segmentation_pane(self, mask, color):
        '''
        Masks sizing, selection and storing functionality
        '''
        mask_widget_plot = QWidget()
        mask_layout = QVBoxLayout()
        overlayed_pixmap = self._overlay_mask_over_content(mask, color, 109)
    
        mask_label = QLabel()
        mask_label.setPixmap(overlayed_pixmap.scaled(512, 512, Qt.KeepAspectRatio))
        mask_layout.addWidget(mask_label)

        radio_selection_button = QRadioButton()
        mask_layout.addWidget(radio_selection_button)

        mask_widget_plot.setLayout(mask_layout)
        
        selection = {"widget": mask_widget_plot, 
                "radio": radio_selection_button}
        
        return selection

    def _overlay_mask_over_content(self, mask, color, intensity, content_pixmap = None):
        '''
        Creating a semi-transparent overlay over content 
        image for a single mask
        '''
        mask = mask.astype(np.uint8) * intensity
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)  
        colored_mask[..., :3] = color 
        colored_mask[..., 3] = mask 
        mask_image = QImage(colored_mask.data, colored_mask.shape[1], colored_mask.shape[0], QImage.Format_RGBA8888)
        mask_pixmap = QPixmap.fromImage(mask_image)

        if content_pixmap is None:
            content_pixmap = QPixmap(self.content_image_path)
        
        composition_painter = QPainter(content_pixmap)
        composition_painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        composition_painter.drawPixmap(0, 0, mask_pixmap)
        composition_painter.end()

        return content_pixmap

    def confirm_and_discard_buttons(self, num_masks):
        '''
        Buttons responsible for selecting or discarding presented masks within the 
        Segmentation (Mask Creation) Pane
        '''
        self.confirmButton = QPushButton("Confirm Selection")
        self.confirmButton.clicked.connect(self._add_selected_mask)
        self.confirmButton.setFont(QFont('Arial', 11, QFont.Bold)) 
        self.maskSegmentationLayout.addWidget(self.confirmButton, 1, num_masks)

        self.discardButton = QPushButton("Discard")
        self.discardButton.clicked.connect(self._remove_segmentation_pane)
        self.discardButton.clicked.connect(self.clear_annotations)
        self.discardButton.setFont(QFont('Arial', 11, QFont.Bold))  
        self.maskSegmentationLayout.addWidget(self.discardButton, 1, num_masks + 1)

    def _clear_mask_space_segmentation_pane(self):
        '''
        Clears previously displayed masks in the Segmentation (Mask Creation) Pane
        '''
        self.layout.removeWidget(self.maskWidget)
        self.maskWidget.deleteLater()

        self.maskWidget = QWidget()
        self.maskWidget.setLayout(self.maskSegmentationLayout)
        self.layout.addWidget(self.maskWidget)

    def _add_selected_mask(self):
        '''
        Saves the user-selected binary mask from the option of three earlier presented, 
        upon the press of the Confirmation button in the Segmentation (Mask Creation) Pane
        '''
        button = self.maskButtonConfirmDiscard.checkedButton()
        if button is None: 
            QMessageBox.warning(self, "No Selection", 
                                "No mask has been selected. Please select a mask before continuing.")
            return 
        
        radio_id = self.maskButtonConfirmDiscard.id(button)
        
        # Re-do feature funcitionlity 
        if self.redo_mode:
            self.user_selected_finalized_masks[self.redo_mask_id] = self.segmentation_pane_masks[radio_id]
            self.redo_mode = False
            self.statusLabel.setText("Normal mode")

            button = self.redo_buttons[self.redo_mask_id]
            button.setStyleSheet("") 
            self.canvasTitleLabel.setText("Interactive Canvas")

            QMessageBox.information(self, "Segmentation Mode Changed", 
                                    "You have exited Re-do mode and returned to Normal mode")
        else:
            self.user_selected_finalized_masks[self.segmentation_pane_selected_mask_id] = self.segmentation_pane_masks[radio_id]
            self.segmentation_pane_selected_mask_id += 1

        self._remove_segmentation_pane()  
        self.clear_annotations()
        self.display_selected_masks()

    def _remove_segmentation_pane(self):
        '''
        Completly clears and removes the Segmentation (Mask Creation) 
        Pane from the interface 
        '''
        if self.maskWidget.layout() is not None:
            current_segmentation_pane = self.maskWidget.layout()
            length_segmentation_pane = range(current_segmentation_pane.count())
            for mask_position in reversed(length_segmentation_pane):
                pane = current_segmentation_pane.itemAt(mask_position)
                pane_widget = pane.widget()
                pane_widget.setParent(None)

        self.maskWidget = QWidget()
        self.layout.addWidget(self.maskWidget)

    def display_selected_masks(self):
        '''
        Layout for the display of confirmed and selected masks 
        in the Mask Display pane
        '''
        if not self.user_selected_finalized_masks:
            QMessageBox.warning(self, "No Selected Masks", 
                                "No masks have been selected yet. Please select a mask or discard the current segmentation")
            return

        self._clear_mask_display_pane() 

        self.selectedDisplayMaskLayout = QGridLayout()
        cumulative_pixmap = QPixmap(self.content_pixmap)

        title_label = QLabel("Individual Masks Display")
        title_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.selectedDisplayMaskLayout.addWidget(title_label, 1, 0, 1, -1) 

        # Individual Masks Visualizations
        np.random.seed(6772)
        for idx, mask in enumerate(self.user_selected_finalized_masks.values()):
            color = np.random.randint(0, 256, 3).tolist()
            cumulative_pixmap = self._overlay_mask_over_content(mask, color, 120, cumulative_pixmap)
            
            individual_pixmap = QPixmap(self.content_pixmap)
            individual_pixmap = self._overlay_mask_over_content(mask, color, 120, individual_pixmap)
            
            mask_widget = self._create_individual_mask_plot(individual_pixmap, idx)
            self.selectedDisplayMaskLayout.addWidget(mask_widget, 2, idx)

        # Cumulitive Masks Visualizations
        cumulative_mask_widget = self._create_cumulative_mask_plot(cumulative_pixmap)
        self.selectedDisplayMaskLayout.addWidget(cumulative_mask_widget, 0, 0) 

        self.selectedMaskWidget = QWidget()
        self.selectedMaskWidget.setLayout(self.selectedDisplayMaskLayout)
        self.layout.addWidget(self.selectedMaskWidget) 

    def _create_cumulative_mask_plot(self, pixmap):
        '''
        Cumulitive Mask Plotting in the Mask Display Pane
        '''
        main_title_label = QLabel("Masks Display")
        main_title_label.setFont(QFont('Arial', 14, QFont.Bold)) 

        title_label = QLabel("Cumulative Masks Display")
        title_label.setFont(QFont('Arial', 12, QFont.Bold))

        mask_label = QLabel()
        mask_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio)) 

        mask_widget_plot = QWidget()
        mask_layout = QVBoxLayout()

        mask_layout.addWidget(main_title_label) 
        mask_layout.addWidget(title_label)
        mask_layout.addWidget(mask_label)
        mask_widget_plot.setLayout(mask_layout)

        return mask_widget_plot

    def _create_individual_mask_plot(self, pixmap, idx):
        '''
        Individual Mask Plotting in the Mask Display Pane and 
        the respective "Re-do" properties 
        '''
        mask_label = QLabel()
        mask_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

        # Add the re-do option beneath each individual mask
        redo_button = QPushButton("Re-do Mask")
        redo_button.setFont(QFont('Arial', 11, QFont.Bold))
        redo_button.clicked.connect(functools.partial(self._redo_mask, idx))

        # Store the button at its respective index 
        self.redo_buttons[idx] = redo_button

        mask_widget_plot = QWidget()
        mask_layout = QVBoxLayout() 
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(redo_button)
        mask_widget_plot.setLayout(mask_layout)

        return mask_widget_plot
        
    def _redo_mask(self, mask_id):
        '''
        Re-do Functionality and Layout
        '''
        self.canvasTitleLabel.setText("Interactive Canvas - Redo Mode")

        button = self.redo_buttons[mask_id]
        button.setStyleSheet("background-color: red;")

        self.clear_annotations()
        self.redo_mode = True
        self.redo_mask_id = mask_id
        self.statusLabel.setText("Redo mode for mask number: {}".format(mask_id))
        QMessageBox.information(self, "Mode Change", 
                                "You have entered re-do mode for mask number {}".format(mask_id))   
     
    def _clear_mask_display_pane(self):
        '''
        Clear the Mask Display Pane of both individual and cumulutive masks,
        preparing the interface for the Style Transfer Pane
        '''
        if getattr(self, 'selectedMaskWidget', None):
            self.layout.removeWidget(self.selectedMaskWidget)
            self.selectedMaskWidget.deleteLater()
            self.selectedMaskWidget = None

    def style_transfer_pane_initiliazation(self):
        '''
        Plotting the layout of the Style Transfer Pane 
        with its respective Individual and Global hyperparamater, 
        upon the press of "I have finished segmenting" button
        '''
        if not self.user_selected_finalized_masks: 
            QMessageBox.warning(self, "No Masks Selected", 
                                "No masks have been selected. Please generate at least one mask before proceeding to style transfer.")
            return  
        
        self._clear_mask_display_pane()

        self.mainVerticalLayout = QVBoxLayout()
        # Title & Sub-title
        style_transfer_title_label = QLabel("Style Transfer")
        style_transfer_title_label.setFont(QFont('Arial', 14, QFont.Bold))
        style_transfer_title_label.setAlignment(Qt.AlignLeft)
        self.mainVerticalLayout.addWidget(style_transfer_title_label)

        local_style_transfer_label = QLabel("Local Style Transfer Adjustments")
        local_style_transfer_label.setFont(QFont('Arial', 13, QFont.Bold)) 
        local_style_transfer_label.setAlignment(Qt.AlignLeft)
        self.mainVerticalLayout.addWidget(local_style_transfer_label)

        self.finalMaskLayout = QGridLayout()
        self.mainVerticalLayout.addLayout(self.finalMaskLayout)
        self.finalMaskWidget = QWidget()
        self.finalMaskWidget.setLayout(self.mainVerticalLayout)
        self.layout.addWidget(self.finalMaskWidget)

        # Display Individual Global and Local Hyperparameters functionalities
        self._individual_masks_plotting_and_properties()
        self._setup_interpolation_weights()
        self._setup_style_strength_slider()
        self._setup_additional_color_properties_adjustments()
        self._setup_keep_color()
        self._setup_final_style_transfer()
        
        # Disable SAM group buttons 
        self.undoButton.setEnabled(False)
        self.clearSelectionButton.setEnabled(False)
        self.segmentButton.setEnabled(False)
        self.finishedSegmentButton.setEnabled(False)

    def _individual_masks_plotting_and_properties(self):
        '''
        Plotting the individual semi-transperant masks over the content image, 
        along with their respecitve individual (local) hyperparameters
        '''
        np.random.seed(6772)
        mask_idx = 0
        for idx, mask in self.user_selected_finalized_masks.items():
            mask_widget = QWidget()
            mask_layout = QVBoxLayout()

            # Individual Masks Visualization as Semi-transeprant Overlay
            color = np.random.randint(0, 256, 3).tolist()  
            content_pixmap = self._overlay_mask_over_content(mask, color, 150, QPixmap(self.content_pixmap))

            mask_label = QLabel()
            mask_label.setFixedSize(300, 300)
            mask_label.setPixmap(content_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            mask_layout.addWidget(mask_label)
                
            # Buttons regarding the modification of an object's HSV color distribution 
            # The toggle and input boxes will both appear, only once all masks have received 
            # a style image    
            toggle_color_button = QCheckBox("Change the object's color palette")
            toggle_color_button.setFont(QFont('Arial', 10, QFont.Bold))
            toggle_color_button.setVisible(False) 
            mask_layout.addWidget(toggle_color_button)
            self.maskColorToggleButtons[idx] = toggle_color_button
            toggle_color_button.stateChanged.connect(self._update_color_adjustments_state)
            
            hex_color_input = QLineEdit()
            hex_color_input.setPlaceholderText("Apply hex color code (format: #RRGGBB)")
            hex_color_input.setFont(QFont('Arial', 10, QFont.Bold))
            hex_color_input.setVisible(False)
            mask_layout.addWidget(hex_color_input)
            self.hexColorInputs[idx] = hex_color_input

            upload_button = QPushButton("Upload Style Image")
            upload_button.setFont(QFont('Arial', 11, QFont.Bold))
            upload_button.clicked.connect(partial(self._upload_style_image_for_mask, idx))
            mask_layout.addWidget(upload_button)

            mask_layout.setSpacing(0)
            mask_widget.setLayout(mask_layout)
            self.finalMaskLayout.addWidget(mask_widget, 0, mask_idx)
            mask_idx += 1

    def _upload_style_image_for_mask(self, mask_id):
        '''
        Upload Style Image button functionality, along with the  
        color palette modification feature set, given that the condition 
        of all masks have received a style image is met 
        '''
        style_image_path, _ = QFileDialog.getOpenFileName(self, "Select a Style Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if not style_image_path:
            return

        style_pixmap = QPixmap(style_image_path)
        scaled_pixmap = style_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.style_uploaded_images[mask_id] = scaled_pixmap

        column_index = mask_id + self.finalMaskLayout.columnCount() - len(self.user_selected_finalized_masks)
        style_widget = None
        for idx in range(self.finalMaskLayout.rowCount()):
            mask_pos = self.finalMaskLayout.itemAtPosition(idx, column_index)
            if mask_pos and isinstance(mask_pos.widget(), QWidget):
                style_widget = mask_pos.widget()
                break
            
        # If a style widget is found procceed forward
        if style_widget:
            style_layout = style_widget.layout()
            for idx in range(style_layout.count()):
                style_img = style_layout.itemAt(idx)
                if isinstance(style_img.widget(), QPushButton):
                    if idx + 1 < style_layout.count():
                        label_item = style_layout.itemAt(idx + 1)
                        if isinstance(label_item.widget(), QLabel):
                            label_item.widget().setPixmap(scaled_pixmap)
                    else:
                        style_img_label = QLabel()
                        style_img_label.setFixedSize(300, 300)
                        style_img_label.setPixmap(scaled_pixmap)
                        style_layout.addWidget(style_img_label)

        self.masks_with_uploaded_styles.add(mask_id)
        if len(self.masks_with_uploaded_styles) == len(self.user_selected_finalized_masks):
            for mask_id in self.user_selected_finalized_masks.keys():
                self.maskColorToggleButtons[mask_id].setVisible(True)
                self.hexColorInputs[mask_id].setVisible(True)
   
    def _setup_interpolation_weights(self):
        '''
        Plotting and functionality of the 
        Interpolation Weights input boxes 
        '''
        self.interpolationWeightsLayout = QGridLayout()
        self.interpolationWeightsLayout.setSpacing(0)

        self.interpolationWeightsInputBoxesValues = []
        for idx in range(len(self.user_selected_finalized_masks)):
            interpolationWeightDspinbox = QDoubleSpinBox()
            interpolationWeightDspinbox.setFont(QFont('Arial', 10, QFont.Bold))
            interpolationWeightDspinbox.setMinimum(0.0)
            interpolationWeightDspinbox.setMaximum(5.0)
            interpolationWeightDspinbox.setSingleStep(0.1)
            interpolationWeightDspinbox.setValue(1.0)

            self.interpolationWeightsInputBoxesValues.append(interpolationWeightDspinbox)
            self.interpolationWeightsLayout.addWidget(interpolationWeightDspinbox, 1, idx, Qt.AlignCenter)

        self.interpolationWeightsLabel = QLabel("Interpolation Weights:")
        self.interpolationWeightsLabel.setFont(QFont('Arial', 11, QFont.Bold))
        self.interpolationWeightsLayout.addWidget(self.interpolationWeightsLabel, 0, 0, 1, len(self.user_selected_finalized_masks), Qt.AlignLeft)
        self.finalMaskLayout.addLayout(self.interpolationWeightsLayout, 2, 0, 1, len(self.user_selected_finalized_masks))

    def _set_interpolation_weights(self):
        '''
        Responsible for setting the interpolation weights 
        in the correct format requierd by the style transfer network 
        '''
        current_locale = locale.getlocale(locale.LC_NUMERIC)
        try:
            locale.setlocale(locale.LC_NUMERIC, 'C')
            self.interpolation_weights = []
            for weight_input in self.interpolationWeightsInputBoxesValues:
                weight_text = weight_input.text().replace(',', '.')
                weight_float = float(weight_text)
                self.interpolation_weights.append(weight_float)

        except ValueError as error:
            print(f"Error in format of interpolation weights: {error}")
            self.interpolation_weights = []

        finally:
            locale.setlocale(locale.LC_NUMERIC, current_locale)

    def _setup_style_strength_slider(self):
        '''
        Plotting and functionality of the 
        Style Strength Slider 
        '''
        # Sub-Title of Global Hyperparameters
        global_style_label = QLabel("Global Style Transfer Adjustments")
        global_style_label.setFont(QFont('Arial', 13, QFont.Bold)) 
        global_style_label.setAlignment(Qt.AlignLeft)
        self.finalMaskLayout.addWidget(global_style_label, 3, 0, 1, len(self.user_selected_finalized_masks))

        self.styleStrengthLayout = QHBoxLayout()
        self.styleStrengthLabel = QLabel("Style Strength:")
        self.styleStrengthLabel.setFont(QFont('Arial', 11, QFont.Bold))
        self.styleStrengthLayout.addWidget(self.styleStrengthLabel)

        # In the thesis, it has explained and showcased the style strenght
        # feautre for values between 0 and 2.

        # However, due to the inability of the Qt slider to have decimal progressions,
        # the style strength slider has been taken to values between 0 and 100 in the interface 
        # (with 50 being respective of 1 as per the thesis proposal).
        self.styleStrengthSlider = QSlider(Qt.Horizontal)
        self.styleStrengthSlider.setMinimum(0)
        self.styleStrengthSlider.setMaximum(100)
        self.styleStrengthSlider.setSingleStep(1)
        self.styleStrengthSlider.setValue(50)
        self.styleStrengthLayout.addWidget(self.styleStrengthSlider)

        self.styleStrengthValueLabel = QLabel(str(self.styleStrengthSlider.value()))
        self.styleStrengthValueLabel.setFont(QFont('Arial', 10, QFont.Bold))
        self.styleStrengthLayout.addWidget(self.styleStrengthValueLabel)

        self.styleStrengthSlider.valueChanged.connect(self._update_style_strength_label)
        self.finalMaskLayout.addLayout(self.styleStrengthLayout, 4, 0, 1, len(self.user_selected_finalized_masks))

    def _update_style_strength_label(self, value):
        '''
        Gets the correct style strength value as per the style transfer 
        network requirments
        '''
        self.styleStrengthValueLabel.setText(str(value))

    def _setup_additional_color_properties_adjustments(self):
        '''
        Plotting and functionality of the additional DoubleSpinBoxes, 
        that allow the Value and Saturation factors
        from HSV color palette to be additionally adapted
        '''
        self.valueFactorLayout = QHBoxLayout()
        self.valueFactorLabel = QLabel("Value Factor:")
        self.valueFactorLabel.setFont(QFont('Arial', 11, QFont.Bold))
        self.valueFactorLayout.addWidget(self.valueFactorLabel)
        self.valueFactorDspinbox = QDoubleSpinBox()
        self.valueFactorDspinbox.setFont(QFont('Arial', 10, QFont.Bold))
        self.valueFactorDspinbox.setMinimum(0.0)
        self.valueFactorDspinbox.setMaximum(1.0)
        self.valueFactorDspinbox.setSingleStep(0.1)
        self.valueFactorDspinbox.setValue(0.4)
        self.valueFactorDspinbox.setEnabled(False)
        self.valueFactorLayout.addWidget(self.valueFactorDspinbox)
        self.finalMaskLayout.addLayout(self.valueFactorLayout, 5, 0, 1, len(self.user_selected_finalized_masks))
        
        self.satFactorLayout = QHBoxLayout()
        self.satFactorLabel = QLabel("Saturation Factor:")
        self.satFactorLabel.setFont(QFont('Arial', 11, QFont.Bold))
        self.satFactorLayout.addWidget(self.satFactorLabel)
        self.satFactorDspinbox = QDoubleSpinBox()
        self.satFactorDspinbox.setFont(QFont('Arial', 10, QFont.Bold))
        self.satFactorDspinbox.setMinimum(0.0)
        self.satFactorDspinbox.setMaximum(5.0)
        self.satFactorDspinbox.setSingleStep(0.1)
        self.satFactorDspinbox.setValue(0.0)
        self.satFactorDspinbox.setEnabled(False)
        self.satFactorLayout.addWidget(self.satFactorDspinbox)
        self.finalMaskLayout.addLayout(self.satFactorLayout, 6, 0, 1, len(self.user_selected_finalized_masks))
        
    def _update_color_adjustments_state(self):
        '''
        Update the status (interaction ability) of the Saturation and Value DoubleSpinBoxes. 
        Given that an user has checked the ability to change color of an object, 
        those will be made available
        '''
        checked_any_q = False
        for toggle_button in self.maskColorToggleButtons.values():
            if toggle_button.isChecked():
                checked_any_q = True
                break
        self.valueFactorDspinbox.setEnabled(checked_any_q)
        self.satFactorDspinbox.setEnabled(checked_any_q)
 
    def _setup_keep_color(self):
        '''
        Plotting and functionality of the Preserve/ Don't preserve color palette feature
        '''
        self.preserveColorDropdownbox = QComboBox()
        self.preserveColorDropdownbox.setFont(QFont('Arial', 11, QFont.Bold))
        self.preserveColorDropdownbox.addItem("Preserve content image colors")
        self.preserveColorDropdownbox.addItem("Don't preserve content image colors")
        self.preserveColorDropdownbox.currentIndexChanged.connect(self._update_preserve_color)
        self.finalMaskLayout.addWidget(self.preserveColorDropdownbox, 7, 0, 1, len(self.user_selected_finalized_masks))

    def _update_preserve_color(self):
        preserve_color_text_selected = self.preserveColorDropdownbox.currentText()
        self.preserve_content_color = (preserve_color_text_selected == "Preserve content image colors")

    def _setup_final_style_transfer(self):
        '''
        Plotting and functionality of the Style transfer initiator 
        ("Fuse Styles") button
        '''
        self.startStyleTransferButton = QPushButton("Fuse Styles")
        self.startStyleTransferButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.startStyleTransferButton.clicked.connect(self.run_style_transfer)
        self.finalMaskLayout.addWidget(self.startStyleTransferButton, 9, 0, 1, len(self.user_selected_finalized_masks))
        
    def init_style_transfer_network(self):
        '''
        Initiliazation of the adapted AdaIN style transfer network
        '''
        # Device set up for Apple Metal and CPU (CUDA excluded due to potential memory shortrages)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        self.adain_net = modified_AdaIN_network()
        adain_trained = torch.load("AdaIN.pth", device)
        adain_trained['state_dict'] = dict_keys_update(adain_trained['state_dict'])
        self.adain_net.load_state_dict(adain_trained['state_dict'])

    def run_style_transfer(self):
        '''
        Set up style transfer hyperparameters for the modified AdaIN network,
        along with executing style transfer
        '''
        # Adjust from the slider values style strength
        style_strength = self.styleStrengthSlider.value() / 50 

        self._set_interpolation_weights()
        content_path = self.content_image_path

        hex_color_codes = self._get_hex_color_codes()
        if not hex_color_codes:
            return

        value_factor = self.valueFactorDspinbox.value()
        sat_factor = self.satFactorDspinbox.value()

        # Modification of the color distrubution of the content image, given that the feature is utalized 
        content_modified_hsv_array = change_hsv(content_path, self.user_selected_finalized_masks, hex_color_codes, 
                                self._toggled_masks_color_transform(), value_factor, sat_factor)
        content_img = content_loader(content_modified_hsv_array)
        
        # Obtaining user-selected binary masks and their respectivly associated style image paths
        style_paths = [path for _, path in self.style_uploaded_images.items()]
        masks = [mask for _, mask in self.user_selected_finalized_masks.items()]
        if len(style_paths) != len(masks):
            QMessageBox.warning(self, "Mismatch between Masks and Styles", 
                                "The number of styles does not match the number of masks. Please provide a style reference image for each individual mask.")
            return
        
        # Loading the style and mask images
        style_imgs = [style_loader(QPixmap(style_path), 512) for style_path in style_paths]
        mask_imgs = [mask_loader(mask) for mask in masks]

        with torch.no_grad():
            stylized_img = self.adain_net(content_img, style_imgs, style_strength, 
                                        self.interpolation_weights, mask_imgs, self.preserve_content_color)

        self._display_stylized_image(stylized_img)
        self._create_toggle_button()
        if self.toggle_button_before_vs_after:
            self._create_toggle_button()

    def _get_hex_color_codes(self):
        '''
        Responsible for setting the hex color codes to the HSV 
        adapter function, based on the user-provided values.
        '''
        hex_template_pattern = r'^#(?:[0-9a-fA-F]{3}){1,2}$'
        hex_color_codes = []

        for idx in self.hexColorInputs.values():
            hex_color = idx.text().strip()
            if not hex_color:
                hex_color_codes.append(None)
                continue
            if bool(re.match(hex_template_pattern, hex_color)):
                hex_color_codes.append(hex_color)
            else:
                QMessageBox.warning(self, "Invalid Hex Color", 
                                    f"Invalid hex color code: {hex_color}")
                return None

        return hex_color_codes
        
    def _display_stylized_image(self, stylized_img):
        '''
        Convert the final result into Pixmap, that can be displayed within the GUI
        '''
        stylized_qimage = stylized_output_converter(stylized_img)
        stylized_pixmap = QPixmap.fromImage(stylized_qimage)
        self.stylized_pixmap = stylized_pixmap

        width = self.promptable_content.width()
        height = self.promptable_content.height()
        scaled = stylized_pixmap.scaled(width, height, Qt.KeepAspectRatio)

        self.promptable_content.setPixmap(scaled)

    def _toggled_masks_color_transform(self): 
        '''
        Get the masks which are going to undergo a colour adaptation (modification)
        '''
        color_transform_masks = []
        for idx, toggle_button in self.maskColorToggleButtons.items():
            if toggle_button.isChecked():
                color_transform_masks.append(idx)
        return color_transform_masks

    def _create_toggle_button(self):
        '''
        Layout for the button allowing the examination of 
        the "Examine Original Content Image vs Currently 
        Stylized Image" push button that can be pressed
        and depressed 
        '''
        if not self.toggle_button_before_vs_after:
            self.toggleButton = QPushButton("Examine Original Content Image vs Currently Stylized Image")
            self.toggleButton.setFont(QFont('Arial', 12, QFont.Bold))
            self.toggleButton.setCheckable(True)
            self.toggleButton.setChecked(False)
            self.toggleButton.clicked.connect(self._toggle_image)
            self.buttonLayout.addWidget(self.toggleButton)
            self.finalMaskLayout.addWidget(self.toggleButton, 8, 0, 1, len(self.user_selected_finalized_masks))
            self.toggle_button_before_vs_after = True

    def _toggle_image(self):
        '''
        Functionality of the "Examine Original Content Image vs
        Currently Stylized Image" push button
        '''
        width = self.promptable_content.width()
        height = self.promptable_content.height()

        scaled_content = self.content_pixmap.scaled(width, height, Qt.KeepAspectRatio)
        scaled_stylized = self.stylized_pixmap.scaled(width, height, Qt.KeepAspectRatio)

        # Show content 
        if self.toggleButton.isChecked():  
            self.promptable_content.setPixmap(scaled_content)
        # Display AdaIN's outcome
        else:  
            self.promptable_content.setPixmap(scaled_stylized)
               
    def reset_all(self):
        '''
        Functionality  of the Start all over again button 
        '''
        # Reset variables
        self.prompt_points_coordinates = []
        self.click_types = []
        self.segmentation_pane_masks = []
        self.segmentation_pane_selected_mask_id = 0
        self.style_uploaded_images = {}
        self.user_selected_finalized_masks.clear()
        self.promptable_content.clear()
        self.toggle_button_before_vs_after = False

        # Ensure the GUI is cleared from any segmentation or style transfer widgets
        self._remove_segmentation_pane()
        self._clear_mask_display_pane()

        if self.toggleButton:
            self.toggleButton.deleteLater()
            self.toggleButton = None  
        if hasattr(self,'finalMaskWidget'):
            self.finalMaskWidget.deleteLater()

        # Reset button states
        self.resetButton.setEnabled(True)
        self.analyzeImageButton.setEnabled(False)
        self.undoButton.setEnabled(True)
        self.clearSelectionButton.setEnabled(True)
        self.segmentButton.setEnabled(True)
        self.finishedSegmentButton.setEnabled(True)

        # Load the promptable version of the content image in the Interactive Canvas
        width = self.promptable_content.width()
        height = self.promptable_content.height()
        scaled = self.content_pixmap.scaled(width, height, Qt.KeepAspectRatio)
        
        self.promptable_content.setPixmap(scaled)

def main_execution():
    app = QApplication(sys.argv)
    gui = InteractiveStyleTransfer()
    gui.showMaximized()
    gui.show()
    app.exec_()
    return gui

if __name__ == '__main__':
    gui_instance = main_execution()
