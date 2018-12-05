import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import datetime as dt


fig=plt.figure()
axl=fig.add_subplot(1,1,1)

ser=serial.Serial("COM4",115200)
beat=[]
avg_beat=[]
xs=[]
ys=[]
y_range=[30,150]

def mov_avg (mylist):
    N = 3
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            if (moving_ave)<140:
                moving_aves.append(moving_ave)
    return moving_aves



    
while 1:
    while(ser.inWaiting() >0):
        def animate(i, xs, ys):
            # Read temperature (Celsius) from TMP102
            #temp_c = round(tmp102.read_temp(), 2)
            
            dat= round(int(ser.readline()),2)
            
            beat.append(dat)
            if len(beat)>15:
                beat.pop(0)
            
            # Add x and y to lists
            xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            ys.append(dat)
            
            # Limit x and y lists to 20 items
            xs = xs[-20:]
            ys = ys[-20:]

            # Draw x and y lists
            axl.clear()
            axl.plot(xs, ys)

            # Format plot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Heart pulse rate moniter')
            plt.ylabel('Heart Pulse')
            axl.set_ylim(y_range)

        
            print(beat)
            m_beat=mov_avg(mov_avg(beat))
            
            print (m_beat)          

            for i in m_beat:
                if (i>m_beat[(m_beat.index(i))-1] and i>120):
                    plt.text(4,3,"drowsy pulse region",ha='left',fontsize=15,wrap=True)
                    print ('drowsy')
        ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
        plt.show()
                
        
        ##        ab= sum(beat)/len(beat)
        ##        avg_beat.append(ab)
        ##        for i in avg_beat:
        ##            if i>avg_beat[avg_beat.index(i)-1] and i>135:
        ##                print ('drowsy')

        
