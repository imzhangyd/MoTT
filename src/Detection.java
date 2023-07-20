/**
 * A detection is defined as a 3D spatial in a specified frame
 * 
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 * */

public class Detection {
	
	public enum DetectionTypes
	{
		REAL, VIRTUAL
	}
	
	final int t;
	double x;
	double y;
	double z;
	DetectionTypes detectionType = DetectionTypes.REAL;
	
	/**
	 * Create a new detection
	 * @param x x-axis coordinate
	 * @param y y-axis coordinate
	 * @param z z-axis coordinate
	 * @param t frame at which the detection can be found
	 * 
	 * */
	public Detection(double x, double y, double z, int t)
	{
		this.x = x;
		this.y = y;
		this.z = z;
		this.t = t;
	}
	
	/**
	 * @return the frame at which the detection can be found 
	 * */
	public int getT() {
		return t;
	}
	
	/**
	 * @return x-axis coordinate
	 * */
	public double getX()
	{
		return x;
	}

	/**
	 * @return y-axis coordinate
	 * */
	public double getY()
	{
		return y;
	}
	
	/**
	 * @return z-axis coordinate
	 * */
	public double getZ()
	{
		return z;
	}
	
	/**
	 * change the type of detection
	 * @param detectionType the type of the detection
	 * */
	public void setDetectionType(DetectionTypes detectionType)
	{
		this.detectionType = detectionType;
	}
	
	/**
	 * @return the type of this detection
	 * */
	public DetectionTypes getDetectionType()
	{
		return this.detectionType;
	}
}
