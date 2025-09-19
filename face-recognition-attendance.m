%% FACE RECOGNITION ATTENDANCE SYSTEM

clc; clear; close all;

%% Step 1: Load the dataset
faceDatabase = imageDatastore('faces_dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into training & testing
[trainImgs, testImgs] = splitEachLabel(faceDatabase,0.8,'randomized');

%% Step 2: Train using bag of features
bag = bagOfFeatures(trainImgs);
classifier = trainImageCategoryClassifier(trainImgs,bag);

%% Step 3: Evaluate accuracy
confMatrix = evaluate(classifier,testImgs);
accuracy = mean(diag(confMatrix));
fprintf('Model Accuracy: %.2f%%\n',accuracy*100);

%% Step 4: Webcam initialization
cam = webcam; 
disp('Press Ctrl+C to stop webcam');

%% Step 5: Attendance CSV file setup
attendanceFile = 'attendance.csv';
if ~isfile(attendanceFile)
    fid = fopen(attendanceFile,'w');
    fprintf(fid,'Name,DateTime\n');
    fclose(fid);
end

%% Step 6: Real-time face recognition and attendance
faceDetector = vision.CascadeObjectDetector; % Face detector

while true
    % Capture frame
    img = snapshot(cam);
    grayImg = rgb2gray(img);
    
    % Detect faces
    bboxes = step(faceDetector,grayImg);
    
    if ~isempty(bboxes)
        for i=1:size(bboxes,1)
            face = imcrop(img,bboxes(i,:));
            
            % Predict label
            [labelIdx, score] = predict(classifier,face);
            name = string(classifier.Labels(labelIdx));
            
            % Display rectangle + name
            img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),name,'Color','green');
            
            % Mark attendance
            T = readtable(attendanceFile);
            nowTime = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
            % check if already marked in this session
            if ~any(strcmp(T.Name,name))
                fid = fopen(attendanceFile,'a');
                fprintf(fid,'%s,%s\n',name,nowTime);
                fclose(fid);
                disp(['Attendance marked for: ', name]);
            end
        end
    end
    
    % Show frame
    imshow(img);
    title('Face Recognition Attendance System');
    drawnow;
end