---
sidebar_position: 5
---

# Unity Robotics Simulation for Humanoid Applications

## Overview

Unity Robotics provides a unique approach to robotics simulation, combining Unity's powerful game engine with robotics-specific tools. Unlike Gazebo and Isaac Sim, Unity excels in creating immersive, visually-rich environments that are particularly well-suited for human-robot interaction research, social robotics, and educational applications.

## Architecture and Integration

### ROS-TCP-Connector Integration

Unity Robotics uses the ROS-TCP-Connector to enable communication between Unity and ROS/ROS2 systems:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using System.Collections.Generic;

public class UnityRobotController : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Robot Configuration")]
    public string robotName = "UnityHumanoid";
    public float controlFrequency = 100.0f; // Hz
    public float simulationSpeed = 1.0f;
    
    // Robot joint control
    [Header("Joint Configuration")]
    public List<JointController> jointControllers;
    
    // Sensor simulation
    [Header("Sensor Configuration")]
    public List<SensorController> sensorControllers;
    
    private ROSConnection rosConnection;
    private float controlTimer = 0f;
    private float controlInterval;
    
    void Start()
    {
        // Initialize ROS connection
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.Initialize(rosIPAddress, rosPort);
        
        // Calculate control interval based on frequency
        controlInterval = 1.0f / controlFrequency;
        
        // Subscribe to ROS topics
        SubscribeToTopics();
        
        // Publish initial state
        InvokeRepeating("PublishRobotState", 0.0f, 0.01f); // 100 Hz for state publishing
    }
    
    void SubscribeToTopics()
    {
        // Subscribe to joint commands
        rosConnection.Subscribe<sensor_msgs.JointState>(
            topic: $"/{robotName}/joint_commands",
            callback: OnJointCommandReceived
        );
        
        // Subscribe to velocity commands
        rosConnection.Subscribe<geometry_msgs.Twist>(
            topic: $"/{robotName}/cmd_vel",
            callback: OnVelocityCommandReceived
        );
        
        // Subscribe to other robot commands (grips, etc.)
        rosConnection.Subscribe<std_msgs.String>(
            topic: $"/{robotName}/behavior_command",
            callback: OnBehaviorCommandReceived
        );
    }
    
    void Update()
    {
        // Execute control loop at specified frequency
        controlTimer += Time.deltaTime;
        if (controlTimer >= controlInterval)
        {
            ExecuteControlLoop();
            controlTimer = 0f;
        }
    }
    
    void ExecuteControlLoop()
    {
        // Apply current joint commands to Unity articulation bodies
        foreach (var jointCtrl in jointControllers)
        {
            if (jointCtrl.articulationBody != null)
            {
                ApplyJointControl(jointCtrl);
            }
        }
        
        // Process sensor data and publish to ROS
        foreach (var sensorCtrl in sensorControllers)
        {
            sensorCtrl.UpdateAndPublish();
        }
    }
    
    void OnJointCommandReceived(sensor_msgs.JointState jointStateMsg)
    {
        // Process incoming joint commands
        for (int i = 0; i < jointStateMsg.name.Count; i++)
        {
            string jointName = jointStateMsg.name[i];
            
            if (i < jointStateMsg.position.Count)
            {
                float targetPosition = (float)jointStateMsg.position[i];
                
                // Find corresponding joint controller
                var jointCtrl = jointControllers.Find(jc => jc.jointName == jointName);
                if (jointCtrl != null)
                {
                    jointCtrl.targetPosition = targetPosition;
                }
            }
        }
    }
    
    void OnVelocityCommandReceived(geometry_msgs.Twist twistMsg)
    {
        // Handle velocity commands for base movement
        // This would typically affect the root/body of the humanoid
        Vector3 linearVel = new Vector3(
            (float)twistMsg.linear.x,
            (float)twistMsg.linear.y, 
            (float)twistMsg.linear.z
        );
        
        Vector3 angularVel = new Vector3(
            (float)twistMsg.angular.x,
            (float)twistMsg.angular.y,
            (float)twistMsg.angular.z
        );
        
        // Apply movement to robot base
        ApplyBaseMovement(linearVel, angularVel);
    }
    
    void OnBehaviorCommandReceived(std_msgs.String behaviorMsg)
    {
        // Handle behavior commands (greeting, waving, etc.)
        StartCoroutine(ExecuteBehavior(behaviorMsg.data));
    }
    
    void ApplyJointControl(JointController jointCtrl)
    {
        if (jointCtrl.articulationBody == null) return;
        
        // Get current joint state
        ArticulationReducedSpacePosition currentPos = jointCtrl.articulationBody.jointPosition;
        ArticulationReducedSpaceVelocity currentVel = jointCtrl.articulationBody.jointVelocity;
        
        // Calculate control action using PD controller
        float positionError = jointCtrl.targetPosition - currentPos.x;
        float velocityError = 0 - currentVel.x; // Assuming target velocity is 0
        
        float controlAction = jointCtrl.kp * positionError + jointCtrl.kd * velocityError;
        
        // Apply control (with safety limits)
        controlAction = Mathf.Clamp(controlAction, -jointCtrl.maxForce, jointCtrl.maxForce);
        
        // Apply force/torque to joint
        ArticulationDrive drive = jointCtrl.articulationBody.xDrive;
        drive.target = jointCtrl.targetPosition * Mathf.Rad2Deg; // Convert to degrees for Unity
        drive.stiffness = jointCtrl.kp;
        drive.damping = jointCtrl.kd;
        jointCtrl.articulationBody.xDrive = drive;
    }
    
    void ApplyBaseMovement(Vector3 linearVel, Vector3 angularVel)
    {
        // Apply movement to the robot's root body
        // This would be the main body/hip of the humanoid robot
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = linearVel;
            rb.angularVelocity = angularVel;
        }
    }
    
    System.Collections.IEnumerator ExecuteBehavior(string behavior)
    {
        // Execute predefined behaviors
        switch (behavior.ToLower())
        {
            case "wave":
                yield return StartCoroutine(ExecuteWaveBehavior());
                break;
            case "greet":
                yield return StartCoroutine(ExecuteGreetBehavior());
                break;
            case "dance":
                yield return StartCoroutine(ExecuteDanceBehavior());
                break;
            default:
                Debug.Log($"Unknown behavior: {behavior}");
                break;
        }
    }
    
    System.Collections.IEnumerator ExecuteWaveBehavior()
    {
        // Example: Wave the right arm
        var armJoint = jointControllers.Find(jc => jc.jointName == "right_shoulder_pitch");
        if (armJoint != null)
        {
            float originalPosition = armJoint.targetPosition;
            
            // Move arm up
            for (float t = 0; t < 1.0f; t += Time.deltaTime * 2.0f)
            {
                armJoint.targetPosition = Mathf.Lerp(originalPosition, 1.0f, t);
                yield return null;
            }
            
            // Wave motion
            for (int i = 0; i < 3; i++)
            {
                // Wave up
                for (float t = 0; t < 0.5f; t += Time.deltaTime * 4.0f)
                {
                    armJoint.targetPosition = 1.0f + Mathf.Sin(t * Mathf.PI * 4) * 0.2f;
                    yield return null;
                }
                
                // Wave down
                for (float t = 0; t < 0.5f; t += Time.deltaTime * 4.0f)
                {
                    armJoint.targetPosition = 1.0f - Mathf.Sin(t * Mathf.PI * 4) * 0.2f;
                    yield return null;
                }
            }
            
            // Return to original position
            for (float t = 0; t < 1.0f; t += Time.deltaTime * 2.0f)
            {
                armJoint.targetPosition = Mathf.Lerp(1.0f, originalPosition, t);
                yield return null;
            }
        }
    }
    
    void PublishRobotState()
    {
        // Create and publish joint state message
        var jointStateMsg = new sensor_msgs.JointState();
        jointStateMsg.header = new std_msgs.Header();
        jointStateMsg.header.stamp = new TimeStamp(rosConnection.GetServerTime());
        jointStateMsg.header.frame_id = "base_link";
        
        foreach (var jointCtrl in jointControllers)
        {
            if (jointCtrl.articulationBody != null)
            {
                jointStateMsg.name.Add(jointCtrl.jointName);
                
                // Get current position in radians
                ArticulationReducedSpacePosition pos = jointCtrl.articulationBody.jointPosition;
                jointStateMsg.position.Add(pos.x);
                
                // Get current velocity
                ArticulationReducedSpaceVelocity vel = jointCtrl.articulationBody.jointVelocity;
                jointStateMsg.velocity.Add(vel.x);
                
                // Optionally add effort (calculated force/torque)
                // jointStateMsg.effort.Add(calculatedEffort);
            }
        }
        
        rosConnection.Publish($"/{robotName}/joint_states", jointStateMsg);
    }
}

[System.Serializable]
public class JointController
{
    public string jointName;
    public ArticulationBody articulationBody;
    [Range(0, 1000)] public float kp = 100f; // Proportional gain
    [Range(0, 200)] public float kd = 20f;   // Derivative gain
    [Range(0, 1000)] public float maxForce = 100f;
    public float targetPosition = 0f;
}

[System.Serializable] 
public class SensorController
{
    public string sensorName;
    public string rosTopic;
    public GameObject sensorObject;
    
    public virtual void UpdateAndPublish()
    {
        // Implementation will vary by sensor type
    }
}
```

## Physics Simulation in Unity

### Articulation Bodies for Humanoid Robots

Unity's Articulation Body system provides a physics-based approach to simulating articulated robots:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class HumanoidPhysicsSetup : MonoBehaviour
{
    [Header("Physical Properties")]
    [Range(0.1f, 10.0f)] public float totalMass = 50.0f;
    public float gravityScale = 1.0f;
    
    [Header("Joint Configuration")]
    public ArticulationDriveConfig leftHipConfig;
    public ArticulationDriveConfig rightHipConfig;
    public ArticulationDriveConfig leftKneeConfig;
    public ArticulationDriveConfig rightKneeConfig;
    public ArticulationDriveConfig leftAnkleConfig;
    public ArticulationDriveConfig rightAnkleConfig;
    
    void Start()
    {
        ConfigureHumanoidPhysics();
    }
    
    void ConfigureHumanoidPhysics()
    {
        // Get all articulation bodies in the robot hierarchy
        ArticulationBody[] allBodies = GetComponentsInChildren<ArticulationBody>();
        
        // Calculate mass distribution based on humanoid body proportions
        AssignMassDistribution(allBodies);
        
        // Configure joints for humanoid movement
        ConfigureJoints(allBodies);
        
        // Set up collision properties
        ConfigureCollisions();
    }
    
    void AssignMassDistribution(ArticulationBody[] bodies)
    {
        // Assign mass based on human body proportions
        float totalCalculatedMass = 0f;
        
        foreach (var body in bodies)
        {
            string bodyName = body.name.ToLower();
            
            // Approximate mass distribution for a 50kg humanoid
            float mass = 0f;
            if (bodyName.Contains("head")) mass = 5.0f;        // 5kg for head
            else if (bodyName.Contains("torso")) mass = 20.0f;  // 20kg for torso
            else if (bodyName.Contains("thigh")) mass = 8.0f;   // 8kg for each thigh
            else if (bodyName.Contains("shank")) mass = 4.0f;   // 4kg for each shank (lower leg)
            else if (bodyName.Contains("foot")) mass = 1.5f;    // 1.5kg for each foot
            else if (bodyName.Contains("upperarm")) mass = 3.0f; // 3kg for each upper arm
            else if (bodyName.Contains("forearm")) mass = 2.0f;  // 2kg for each fore arm
            else mass = 1.0f; // Default mass for other parts
            
            body.mass = mass;
            totalCalculatedMass += mass;
        }
        
        // Scale masses to match desired total mass
        float scale = totalMass / totalCalculatedMass;
        foreach (var body in bodies)
        {
            body.mass *= scale;
        }
    }
    
    void ConfigureJoints(ArticulationBody[] bodies)
    {
        foreach (var body in bodies)
        {
            string bodyName = body.name.ToLower();
            
            // Configure joint drives based on joint type and location
            ArticulationDrive xDrive = body.xDrive;
            ArticulationDrive yDrive = body.yDrive;
            ArticulationDrive zDrive = body.zDrive;
            
            if (bodyName.Contains("hip"))
            {
                ConfigureJointForHip(body, ref xDrive, ref yDrive, ref zDrive);
            }
            else if (bodyName.Contains("knee"))
            {
                ConfigureJointForKnee(body, ref xDrive, ref yDrive, ref zDrive);
            }
            else if (bodyName.Contains("ankle"))
            {
                ConfigureJointForAnkle(body, ref xDrive, ref yDrive, ref zDrive);
            }
            else if (bodyName.Contains("shoulder"))
            {
                ConfigureJointForShoulder(body, ref xDrive, ref yDrive, ref zDrive);
            }
            else if (bodyName.Contains("elbow"))
            {
                ConfigureJointForElbow(body, ref xDrive, ref yDrive, ref zDrive);
            }
            
            body.xDrive = xDrive;
            body.yDrive = yDrive;
            body.zDrive = zDrive;
        }
    }
    
    void ConfigureJointForHip(ArticulationBody body, ref ArticulationDrive x, ref ArticulationDrive y, ref ArticulationDrive z)
    {
        // Hip joint: typically 3 DOF (yaw, pitch, roll)
        ConfigureDrive(ref x, 1000f, 100f, 3.14f, -3.14f); // Yaw
        ConfigureDrive(ref y, 1000f, 100f, 1.57f, -0.5f);  // Pitch (allow forward bend)
        ConfigureDrive(ref z, 800f, 80f, 0.5f, -0.5f);     // Roll (limited)
    }
    
    void ConfigureJointForKnee(ArticulationBody body, ref ArticulationDrive x, ref ArticulationDrive y, ref ArticulationDrive z)
    {
        // Knee joint: primarily flexion
        ConfigureDrive(ref x, 2000f, 200f, 2.5f, 0f);  // Flexion only (0 to ~143 degrees)
        ConfigureDrive(ref y, 0f, 0f, 0f, 0f);          // Locked in other axes
        ConfigureDrive(ref z, 0f, 0f, 0f, 0f);
    }
    
    void ConfigureJointForAnkle(ArticulationBody body, ref ArticulationDrive x, ref ArticulationDrive y, ref ArticulationDrive z)
    {
        // Ankle joint: pitch and limited roll
        ConfigureDrive(ref x, 500f, 50f, 0.5f, -0.5f);  // Pitch (dorsiflexion/plantarflexion)
        ConfigureDrive(ref y, 0f, 0f, 0f, 0f);           // Typically locked
        ConfigureDrive(ref z, 300f, 30f, 0.3f, -0.3f);   // Roll (inversion/eversion)
    }
    
    void ConfigureJointForShoulder(ArticulationBody body, ref ArticulationDrive x, ref ArticulationDrive y, ref ArticulationDrive z)
    {
        // Shoulder joint: 3 DOF with significant range
        ConfigureDrive(ref x, 500f, 50f, 1.57f, -1.57f); // Yaw
        ConfigureDrive(ref y, 500f, 50f, 2.5f, -0.5f);   // Pitch
        ConfigureDrive(ref z, 500f, 50f, 1.57f, -1.57f); // Roll
    }
    
    void ConfigureJointForElbow(ArticulationBody body, ref ArticulationDrive x, ref ArticulationDrive y, ref ArticulationDrive z)
    {
        // Elbow joint: primarily flexion
        ConfigureDrive(ref x, 1000f, 100f, 2.5f, 0f);   // Flexion only
        ConfigureDrive(ref y, 0f, 0f, 0f, 0f);           // Locked
        ConfigureDrive(ref z, 0f, 0f, 0f, 0f);
    }
    
    void ConfigureDrive(ref ArticulationDrive drive, float stiffness, float damping, float upperLimit, float lowerLimit)
    {
        drive.stiffness = stiffness;
        drive.damping = damping;
        drive.forceLimit = 1000f; // Maximum force (N)
        drive.lowerLimit = lowerLimit;
        drive.upperLimit = upperLimit;
        drive.hasLimits = true;
    }
    
    void ConfigureCollisions()
    {
        // Add appropriate colliders to prevent self-intersection
        // This is typically done through Unity editor or automatically via URDF import
    }
}
```

## Perception and Sensor Simulation

### Camera and Vision Sensors

Unity's rendering pipeline can be leveraged for sophisticated vision sensor simulation:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using System.Collections;
using System.Collections.Generic;

public class UnityVisionSensor : SensorController
{
    [Header("Camera Configuration")]
    public Camera sensorCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float updateRate = 30.0f; // Hz
    
    [Header("Sensor Properties")]
    public bool enableDepth = true;
    public bool enableSemanticSegmentation = false;
    public float minDepth = 0.1f;
    public float maxDepth = 10.0f;
    
    private float updateInterval;
    private float lastUpdateTime;
    private RenderTexture renderTexture;
    private ROSConnection rosConnection;
    
    void Start()
    {
        rosConnection = ROSConnection.GetOrCreateInstance();
        
        // Calculate update interval
        updateInterval = 1.0f / updateRate;
        
        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        sensorCamera.targetTexture = renderTexture;
    }
    
    public override void UpdateAndPublish()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishImage();
            lastUpdateTime = Time.time;
        }
    }
    
    void PublishImage()
    {
        // Capture image from camera
        Texture2D image = CaptureImageFromCamera();
        
        // Create ROS Image message
        var rosImage = new sensor_msgs.Image();
        
        // Fill header
        rosImage.header = new std_msgs.Header();
        rosImage.header.stamp = new TimeStamp(rosConnection.GetServerTime());
        rosImage.header.frame_id = sensorName + "_optical_frame";
        
        // Fill image properties
        rosImage.height = (uint)imageHeight;
        rosImage.width = (uint)imageWidth;
        rosImage.encoding = "rgb8"; // For RGB images
        rosImage.is_bigendian = 0;
        rosImage.step = (uint)(imageWidth * 3); // 3 bytes per pixel for RGB
        
        // Convert texture data to byte array
        byte[] imageData = image.GetRawTextureData<byte>();
        rosImage.data = new List<byte>(imageData);
        
        // Publish the image
        rosConnection.Publish(rosTopic, rosImage);
        
        // Clean up
        Destroy(image);
    }
    
    Texture2D CaptureImageFromCamera()
    {
        // Create a temporary RenderTexture to read the camera output
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;
        
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        image.Apply();
        
        RenderTexture.active = currentRT;
        return image;
    }
    
    // Method for depth image capture (simplified)
    public Texture2D CaptureDepthImage()
    {
        // In a real implementation, this would render depth information
        // to the camera and read it as a grayscale image
        Texture2D depthImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RFloat, false);
        
        // This is a placeholder - in practice, you'd render depth to this texture
        // using a custom shader that encodes depth information
        
        return depthImage;
    }
}

public class UnityMultiModalSensor : MonoBehaviour
{
    [Header("Multi-Modal Sensor Setup")]
    public List<UnityVisionSensor> visionSensors;
    public UnityIMUSensor imuSensor;
    public UnityForceTorqueSensor forceTorqueSensor;
    
    [Header("Sensor Fusion Configuration")]
    public bool enableSensorFusion = true;
    
    void Start()
    {
        InitializeSensors();
    }
    
    void InitializeSensors()
    {
        // Initialize all vision sensors
        foreach (var visionSensor in visionSensors)
        {
            visionSensor.enabled = true;
        }
        
        // Initialize other sensors
        if (imuSensor != null)
            imuSensor.enabled = true;
            
        if (forceTorqueSensor != null)
            forceTorqueSensor.enabled = true;
    }
    
    void Update()
    {
        if (enableSensorFusion)
        {
            ProcessSensorFusion();
        }
    }
    
    void ProcessSensorFusion()
    {
        // In a real implementation, this would combine data from multiple sensors
        // to create a more comprehensive perception of the environment
        
        // Example: Combine camera and depth information for 3D object detection
        // Example: Fuse IMU data with vision for better pose estimation
        // Example: Combine force/torque with vision for manipulation planning
    }
}

public class UnityIMUSensor : SensorController
{
    [Header("IMU Configuration")]
    public float updateRate = 200.0f; // Hz
    public Vector3 noiseLevel = new Vector3(0.01f, 0.01f, 0.01f);
    
    private float updateInterval;
    private float lastUpdateTime;
    private Rigidbody attachedRigidbody;
    
    void Start()
    {
        updateInterval = 1.0f / updateRate;
        attachedRigidbody = GetComponent<Rigidbody>();
        if (attachedRigidbody == null)
            attachedRigidbody = GetComponentInParent<Rigidbody>();
    }
    
    public override void UpdateAndPublish()
    {
        if (attachedRigidbody == null) return;
        
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishIMUData();
            lastUpdateTime = Time.time;
        }
    }
    
    void PublishIMUData()
    {
        // Create IMU message
        var imuMsg = new sensor_msgs.Imu();
        imuMsg.header = new std_msgs.Header();
        imuMsg.header.stamp = new TimeStamp(
            Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance().GetServerTime()
        );
        imuMsg.header.frame_id = sensorName + "_link";
        
        // Fill orientation from Unity rotation
        // Convert from Unity coordinates to ROS coordinates
        Quaternion unityRotation = transform.rotation;
        imuMsg.orientation.x = unityRotation.x;
        imuMsg.orientation.y = unityRotation.z;  // Unity Z becomes ROS Y
        imuMsg.orientation.z = unityRotation.y;  // Unity Y becomes ROS Z
        imuMsg.orientation.w = unityRotation.w;
        
        // Fill angular velocity
        Vector3 angularVel = attachedRigidbody.angularVelocity;
        imuMsg.angular_velocity.x = angularVel.x;
        imuMsg.angular_velocity.y = angularVel.z;  // Unity Z becomes ROS Y
        imuMsg.angular_velocity.z = angularVel.y;  // Unity Y becomes ROS Z
        
        // Fill linear acceleration
        Vector3 linearAcc = attachedRigidbody.velocity; // Simplified - should differentiate
        imuMsg.linear_acceleration.x = linearAcc.x;
        imuMsg.linear_acceleration.y = linearAcc.z;  // Unity Z becomes ROS Y
        imuMsg.linear_acceleration.z = linearAcc.y;  // Unity Y becomes ROS Z
        
        // Add noise to simulate real sensor
        AddNoiseToIMU(ref imuMsg);
        
        // Publish IMU data
        var rosConnection = Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance();
        rosConnection.Publish(rosTopic, imuMsg);
    }
    
    void AddNoiseToIMU(ref sensor_msgs.Imu imuMsg)
    {
        System.Random rand = new System.Random();
        
        // Add Gaussian noise to measurements
        float noiseX = (float)(rand.NextDouble() - 0.5) * noiseLevel.x;
        float noiseY = (float)(rand.NextDouble() - 0.5) * noiseLevel.y;
        float noiseZ = (float)(rand.NextDouble() - 0.5) * noiseLevel.z;
        
        imuMsg.orientation.x += noiseX;
        imuMsg.orientation.y += noiseY;
        imuMsg.orientation.z += noiseZ;
        
        imuMsg.angular_velocity.x += noiseX;
        imuMsg.angular_velocity.y += noiseY;
        imuMsg.angular_velocity.z += noiseZ;
        
        imuMsg.linear_acceleration.x += noiseX;
        imuMsg.linear_acceleration.y += noiseY;
        imuMsg.linear_acceleration.z += noiseZ;
    }
}
```

## Human-Robot Interaction Features

### VR/AR Support for Immersive Interfaces

Unity's strength in VR/AR makes it ideal for creating immersive human-robot interfaces:

```csharp
using UnityEngine;
using UnityEngine.XR;

public class HumanRobotInterfaceVR : MonoBehaviour
{
    [Header("VR Configuration")]
    public Transform playerHead;
    public Transform leftController;
    public Transform rightController;
    
    [Header("Interaction Parameters")]
    public float interactionDistance = 3.0f;
    public LayerMask interactionLayers;
    
    [Header("Social Interaction")]
    public float personalSpaceRadius = 1.5f;
    public float attentionDistance = 5.0f;
    
    private UnityRobotController robotController;
    private bool userDetected = false;
    private float lastGreetingTime = 0f;
    private float greetingCooldown = 10.0f; // seconds
    
    void Start()
    {
        robotController = FindObjectOfType<UnityRobotController>();
        
        // Initialize VR components
        InitializeVRTracking();
    }
    
    void Update()
    {
        DetectAndRespondToUser();
        UpdateSocialBehaviors();
    }
    
    void InitializeVRTracking()
    {
        // The playerHead, leftController, and rightController should be
        // set up by your VR/AR SDK (Oculus, OpenXR, etc.)
    }
    
    void DetectAndRespondToUser()
    {
        if (playerHead == null) return;
        
        // Calculate distance to user
        float distanceToUser = Vector3.Distance(transform.position, playerHead.position);
        
        // Update robot behavior based on user proximity
        if (distanceToUser < attentionDistance)
        {
            userDetected = true;
            
            // Trigger greeting if enough time has passed
            if (distanceToUser < interactionDistance && 
                Time.time - lastGreetingTime > greetingCooldown)
            {
                TriggerGreeting();
            }
            
            // Adjust robot behavior based on distance
            AdjustBehaviorForDistance(distanceToUser);
        }
        else
        {
            userDetected = false;
        }
    }
    
    void AdjustBehaviorForDistance(float distance)
    {
        if (robotController == null) return;
        
        if (distance < personalSpaceRadius)
        {
            // If user is in personal space, robot should be more cautious
            var rosConnection = Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance();
            var retreatMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.String();
            retreatMsg.data = "personal_space_respected";
            rosConnection.Publish("/" + robotController.robotName + "/behavior_command", retreatMsg);
        }
        else if (distance < interactionDistance)
        {
            // In interaction range, robot should show attention
            var rosConnection = Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance();
            var attentionMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.String();
            attentionMsg.data = "paying_attention";
            rosConnection.Publish("/" + robotController.robotName + "/behavior_command", attentionMsg);
        }
    }
    
    void TriggerGreeting()
    {
        if (robotController == null) return;
        
        var rosConnection = Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance();
        var greetingMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.String();
        greetingMsg.data = "greeting";
        rosConnection.Publish("/" + robotController.robotName + "/behavior_command", greetingMsg);
        
        lastGreetingTime = Time.time;
    }
    
    void UpdateSocialBehaviors()
    {
        if (!userDetected || playerHead == null) return;
        
        // Make robot look at user
        Vector3 directionToUser = (playerHead.position - transform.position).normalized;
        float angleToUser = Vector3.Angle(transform.forward, directionToUser);
        
        if (angleToUser > 10f) // Only turn if significantly not facing user
        {
            // Rotate robot to face user
            Vector3 targetDirection = new Vector3(directionToUser.x, 0, directionToUser.z).normalized;
            if (targetDirection != Vector3.zero)
            {
                transform.rotation = Quaternion.LookRotation(targetDirection, Vector3.up);
            }
        }
        
        // Handle controller-based interactions
        HandleControllerInteractions();
    }
    
    void HandleControllerInteractions()
    {
        // Handle gestures or pointing from VR controllers
        if (leftController != null && rightController != null)
        {
            // Check if user is pointing at the robot with either controller
            CheckControllerPointing(leftController, "left");
            CheckControllerPointing(rightController, "right");
        }
    }
    
    void CheckControllerPointing(Transform controller, string controllerName)
    {
        RaycastHit hit;
        Vector3 controllerForward = controller.TransformDirection(Vector3.forward);
        
        if (Physics.Raycast(controller.position, controllerForward, out hit, interactionDistance, interactionLayers))
        {
            if (hit.collider.CompareTag("RobotPart") || hit.collider.transform.IsChildOf(transform))
            {
                // User is pointing at robot - trigger attention behavior
                TriggerPointedAtBehavior(controllerName);
            }
        }
    }
    
    void TriggerPointedAtBehavior(string controllerName)
    {
        // Robot acknowledges being pointed at
        var rosConnection = Unity.Robotics.ROSTCPConnector.ROSConnection.GetOrCreateInstance();
        var acknowledgmentMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.String();
        acknowledgmentMsg.data = $"acknowledged_pointing_from_{controllerName}";
        rosConnection.Publish("/" + robotController.robotName + "/behavior_command", acknowledgmentMsg);
    }
}

public class SocialBehaviorController : MonoBehaviour
{
    [Header("Behavior Configuration")]
    public float attentionThreshold = 5.0f;
    public float interactionDistance = 2.0f;
    public float personalSpaceDistance = 1.5f;
    
    [Header("Animation Triggers")]
    public string attentionAnimation = "pay_attention";
    public string greetingAnimation = "greeting";
    public string retreatAnimation = "step_back";
    public string idleAnimation = "idle";
    
    private Animator animator;
    private bool isInPersonalSpace = false;
    private bool isAttentive = false;
    
    void Start()
    {
        animator = GetComponent<Animator>();
    }
    
    public void UpdateSocialBehavior(Vector3 userPosition)
    {
        float distance = Vector3.Distance(transform.position, userPosition);
        
        // Check personal space violation
        if (distance < personalSpaceDistance && !isInPersonalSpace)
        {
            isInPersonalSpace = true;
            TriggerRetreatBehavior();
        }
        else if (distance > personalSpaceDistance && isInPersonalSpace)
        {
            isInPersonalSpace = false;
            TriggerIdleBehavior();
        }
        
        // Check for attention and interaction
        if (distance < attentionThreshold && distance > personalSpaceDistance)
        {
            if (!isAttentive)
            {
                isAttentive = true;
                TriggerAttentionBehavior();
            }
            
            // Check for close interaction
            if (distance < interactionDistance)
            {
                TriggerGreetingBehavior();
            }
        }
        else if (isAttentive)
        {
            isAttentive = false;
            TriggerIdleBehavior();
        }
    }
    
    void TriggerAttentionBehavior()
    {
        if (animator != null)
        {
            animator.SetBool(attentionAnimation, true);
            animator.SetBool(idleAnimation, false);
        }
    }
    
    void TriggerGreetingBehavior()
    {
        if (animator != null)
        {
            animator.SetTrigger(greetingAnimation);
        }
    }
    
    void TriggerRetreatBehavior()
    {
        if (animator != null)
        {
            animator.SetTrigger(retreatAnimation);
        }
    }
    
    void TriggerIdleBehavior()
    {
        if (animator != null)
        {
            animator.SetBool(attentionAnimation, false);
            animator.SetBool(idleAnimation, true);
        }
    }
}
```

## Performance Optimization

### Multi-Resolution Simulation

Unity allows for multi-resolution simulation approaches to balance quality and performance:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MultiResolutionSimulator : MonoBehaviour
{
    [Header("Simulation Optimization")]
    [Range(0.1f, 1.0f)] public float highDetailRange = 10.0f;
    [Range(10.0f, 100.0f)] public float lowDetailRange = 50.0f;
    
    [Header("Detail Levels")]
    public int highDetailLOD = 0;
    public int mediumDetailLOD = 1;
    public int lowDetailLOD = 2;
    
    private List<SimulatedObject> simulatedObjects = new List<SimulatedObject>();
    private Transform viewerTransform; // Camera or main viewpoint
    
    void Start()
    {
        viewerTransform = Camera.main.transform;
        InitializeSimulatedObjects();
    }
    
    void Update()
    {
        UpdateObjectDetailLevels();
    }
    
    void InitializeSimulatedObjects()
    {
        // Find all objects that participate in simulation
        GameObject[] allObjects = GameObject.FindGameObjectsWithTag("Simulatable");
        
        foreach (GameObject obj in allObjects)
        {
            SimulatedObject simObj = new SimulatedObject();
            simObj.gameObject = obj;
            simObj.originalLOD = 0; // Set to highest detail initially
            simObj.currentLOD = 0;
            simulatedObjects.Add(simObj);
        }
    }
    
    void UpdateObjectDetailLevels()
    {
        if (viewerTransform == null) return;
        
        foreach (SimulatedObject simObj in simulatedObjects)
        {
            if (simObj.gameObject == null) continue;
            
            float distance = Vector3.Distance(viewerTransform.position, simObj.gameObject.transform.position);
            
            int targetLOD;
            if (distance < highDetailRange)
            {
                targetLOD = highDetailLOD;
            }
            else if (distance < lowDetailRange)
            {
                targetLOD = mediumDetailLOD;
            }
            else
            {
                targetLOD = lowDetailLOD;
            }
            
            // Only apply changes if LOD level changed
            if (targetLOD != simObj.currentLOD)
            {
                SetObjectLOD(simObj, targetLOD);
                simObj.currentLOD = targetLOD;
            }
        }
    }
    
    void SetObjectLOD(SimulatedObject simObj, int lodLevel)
    {
        if (simObj.meshRenderer != null)
        {
            // Adjust renderer settings based on LOD
            switch (lodLevel)
            {
                case 0: // High detail
                    simObj.meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
                    simObj.meshRenderer.receiveShadows = true;
                    break;
                case 1: // Medium detail
                    simObj.meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.TwoSided;
                    simObj.meshRenderer.receiveShadows = false;
                    break;
                case 2: // Low detail
                    simObj.meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    simObj.meshRenderer.receiveShadows = false;
                    break;
            }
        }
        
        // Adjust physics simulation if applicable
        if (simObj.articulationBody != null)
        {
            // For distant objects, we might reduce physics update frequency
            // or use simplified collision shapes
        }
    }
}

[System.Serializable]
public class SimulatedObject
{
    public GameObject gameObject;
    public MeshRenderer meshRenderer;
    public ArticulationBody articulationBody;
    public int originalLOD;
    public int currentLOD;
    
    public SimulatedObject()
    {
        // Initialize with default values
        originalLOD = 0;
        currentLOD = 0;
    }
}

public class PhysicsOptimizationManager : MonoBehaviour
{
    [Header("Physics Optimization")]
    public float farPhysicsDistance = 20.0f;
    public float simulationSpeedScale = 1.0f;
    
    private List<ArticulationBody> farArticulationBodies = new List<ArticulationBody>();
    
    void Start()
    {
        // Find all articulation bodies in the scene
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        
        foreach (ArticulationBody body in allBodies)
        {
            if (Vector3.Distance(transform.position, body.transform.position) > farPhysicsDistance)
            {
                farArticulationBodies.Add(body);
            }
        }
    }
    
    void Update()
    {
        // Reduce simulation quality for distant objects
        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            foreach (ArticulationBody body in farArticulationBodies)
            {
                float distanceToCamera = Vector3.Distance(mainCam.transform.position, body.transform.position);
                
                if (distanceToCamera > farPhysicsDistance)
                {
                    // Reduce solver iterations for distant bodies
                    // This requires Unity Physics package customization
                    // For now, this is conceptual
                    ReducePhysicsQualityForBody(body, distanceToCamera);
                }
            }
        }
    }
    
    void ReducePhysicsQualityForBody(ArticulationBody body, float distance)
    {
        // Conceptual implementation for reducing physics quality
        // Based on distance from camera
        float qualityFactor = Mathf.Clamp01(1.0f - (distance - farPhysicsDistance) / 10.0f);
        
        // In a real implementation, you would adjust:
        // - Solver iterations
        // - Collision detection frequency
        // - Joint constraint quality
        // This requires a custom physics pipeline
    }
}
```

## Educational Applications

### Teaching Humanoid Robotics Concepts

Unity's visual nature makes it excellent for educational purposes:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class EducationalHumanoidInterface : MonoBehaviour
{
    [Header("Educational UI")]
    public Text robotStatusText;
    public Text jointStatusText;
    public Text lessonStatusText;
    public Text physicsExplanationText;
    public Button[] lessonButtons;
    
    [Header("Educational Content")]
    public EducationalLesson[] lessons;
    public int currentLesson = 0;
    
    private UnityRobotController robotController;
    private UnityPhysicsSetup physicsSetup;
    
    void Start()
    {
        robotController = FindObjectOfType<UnityRobotController>();
        physicsSetup = FindObjectOfType<UnityPhysicsSetup>();
        InitializeLessonInterface();
        LoadLesson(currentLesson);
    }
    
    void InitializeLessonInterface()
    {
        // Setup lesson buttons
        for (int i = 0; i < lessonButtons.Length && i < lessons.Length; i++)
        {
            int lessonIndex = i; // Capture for closure
            lessonButtons[i].onClick.AddListener(() => LoadLesson(lessonIndex));
            lessonButtons[i].GetComponentInChildren<Text>().text = lessons[i].title;
        }
    }
    
    public void LoadLesson(int lessonIndex)
    {
        if (lessonIndex >= 0 && lessonIndex < lessons.Length)
        {
            EducationalLesson lesson = lessons[lessonIndex];
            currentLesson = lessonIndex;
            
            // Update UI
            lessonStatusText.text = $"Lesson {lessonIndex + 1}: {lesson.title}";
            physicsExplanationText.text = lesson.physicsExplanation;
            
            // Configure robot for this lesson
            ConfigureRobotForLesson(lesson);
            
            // Show lesson-specific controls
            ShowLessonControls(lesson);
        }
    }
    
    void ConfigureRobotForLesson(EducationalLesson lesson)
    {
        // Configure the robot based on lesson requirements
        switch (lesson.topic)
        {
            case LessonTopic.Balance:
                ConfigureBalanceLesson();
                break;
            case LessonTopic.Walking:
                ConfigureWalkingLesson();
                break;
            case LessonTopic.Manipulation:
                ConfigureManipulationLesson();
                break;
            case LessonTopic.Kinematics:
                ConfigureKinematicsLesson();
                break;
        }
    }
    
    void ConfigureBalanceLesson()
    {
        // Setup robot in balance scenario
        // Add balance challenges, visual aids, etc.
    }
    
    void ConfigureWalkingLesson()
    {
        // Setup robot in walking scenario
        // Add terrain challenges, walking patterns, etc.
    }
    
    void ConfigureManipulationLesson()
    {
        // Setup robot in manipulation scenario
        // Add objects to manipulate, challenges, etc.
    }
    
    void ConfigureKinematicsLesson()
    {
        // Setup robot for kinematics demonstration
        // Add joint controls, visualization, etc.
    }
    
    void ShowLessonControls(EducationalLesson lesson)
    {
        // Show/hide controls based on lesson type
        // This would update the UI to show relevant controls
    }
    
    void Update()
    {
        // Update status displays
        UpdateRobotStatus();
    }
    
    void UpdateRobotStatus()
    {
        if (robotController != null)
        {
            // Update status text with current robot information
            if (robotController.jointControllers != null)
            {
                string jointInfo = "";
                foreach (var jointCtrl in robotController.jointControllers)
                {
                    if (jointCtrl.articulationBody != null)
                    {
                        jointInfo += $"{jointCtrl.jointName}: {jointCtrl.targetPosition:F2} ";
                    }
                }
                jointStatusText.text = "Joints: " + jointInfo;
            }
            
            // Update other status information
            robotStatusText.text = "Status: Running";
        }
    }
    
    public void ExecuteLessonTask(string task)
    {
        // Execute a specific task related to the current lesson
        switch (task)
        {
            case "balance_challenge":
                // Execute balance challenge
                break;
            case "walk_forward":
                // Execute walking command
                break;
            case "wave_hello":
                // Execute greeting behavior
                StartCoroutine(robotController.ExecuteBehavior("wave"));
                break;
        }
    }
}

[System.Serializable]
public class EducationalLesson
{
    public string title;
    public LessonTopic topic;
    public string physicsExplanation;
    public string[] learningObjectives;
    public string[] requiredEquipment;
    public LessonActivity[] activities;
}

public enum LessonTopic
{
    Balance,
    Walking,
    Manipulation,
    Kinematics,
    Dynamics,
    Control,
    Perception,
    SocialInteraction
}

[System.Serializable]
public class LessonActivity
{
    public string activityTitle;
    public string instructions;
    public string successCriteria;
    public float timeEstimate;
}

// Example lesson definition
public class ExampleLesson : MonoBehaviour
{
    void Start()
    {
        // This would define the actual lesson content
        EducationalLesson balanceLesson = new EducationalLesson
        {
            title = "Understanding Balance in Humanoid Robots",
            topic = LessonTopic.Balance,
            physicsExplanation = "Balance in humanoid robots is maintained by keeping the center of mass over the support polygon formed by the feet...",
            learningObjectives = new string[] {
                "Understand the concept of center of mass",
                "Learn how joint control affects balance",
                "Experience the challenge of bipedal balance"
            },
            requiredEquipment = new string[] { "Humanoid robot model", "Balance challenge environment" },
            activities = new LessonActivity[] {
                new LessonActivity {
                    activityTitle = "Center of Mass Visualization",
                    instructions = "Observe how the center of mass moves when you change joint angles",
                    successCriteria = "Student can identify when the robot is balanced vs unbalanced",
                    timeEstimate = 15.0f
                }
            }
        };
    }
}
```

## Hands-on Exercise

1. Using the educational AI agents, design a Unity simulation environment where a humanoid robot learns to navigate around obstacles while maintaining balance.

2. Implement a sensor fusion system that combines data from multiple sensors to improve the robot's perception of its environment.

3. Consider how Unity's VR capabilities could enhance the educational experience for humanoid robotics concepts.

The next section will explore how to integrate simulation results with real robot systems and best practices for sim-to-real transfer in Unity Robotics.