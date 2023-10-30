
import json
import sys
import ray
import shutil
import numpy as np
import glob,os

#@ray.remote

class DataCondenser(object):
    '''
    Info and methods to condense the data into a nice npz file

    attributes
    ----------
    year
    datapath
    fname_base
    minimal_second
    start_day

    methods
    -------
    '''


    def __init__(self,
                 year=2022
                 ):

        self.year = year
        self.datapath = os.path.join(os.getcwd(), 'data', str(year))
        self.fname_base = str(year) + '_place_canvas_history-0000000000'
        self.minimal_second = 46826 if (year==2023) else 45850 #number of seconds before start in the first day
        self.start_day = 20 if (year==2023) else 1 # starting day of the month for the experiment

    def fullfname(self, namebase):
        return os.path.join(self.datapath, namebase)
    
    def fullorigfname(self, namebase):
        return os.path.join(self.datapath, 'original_data', namebase)
        
    def get_key(val,mydict):
        ###### function to return key for any value ##### do not use for long lists, is quite slow
        for key, value in mydict.items():
            if val == value:
                return key
        ValueError("ERROR: key doesn't exist in get_key")

    def textTimestamp_fromSeconds(self, second):
        '''
        Gives the timestamp in the form of the original csv dataset

        parameters
        ----------
        seconds in float (including milliseconds)

        returns
        ----------
        Full text timestamp    
        '''
        seconds = second+self.minimal_second+1e-5
        sec_int = int(seconds)
        ms = int(1000*(seconds-float(sec_int)))
        if ms==0:
            ms_str = ''
        else:
            ms_str = '.'+'{:03d}'.format(ms)
            if ms_str.endswith("0"):
                ms_str = ms_str[:-1]
            if ms_str.endswith("0"):
                ms_str = ms_str[:-1]

        day = sec_int//86400
        hour = (sec_int-day*86400)//3600
        minute = (sec_int-day*86400-hour*3600)//60
        s = sec_int-day*86400-hour*3600-minute*60

        return '2022-04-'+('0' if self.year==2022 else '')+str(self.start_day+day)+' '+ '{:02d}'.format(hour)+':'+ '{:02d}'.format(minute)+':'+ '{:02d}'.format(s)+ms_str+' UTC'
        
    def simplified_timestamp(self, date, hour):
        '''
        Transforms the date and time from the string timestamp into a simplified one, consisting of two ints
        
        parameters
        ----------
        two strings: the date (y-m-d), and the time (h:m:s.ms)
        
        returns
        -------
        one int for the seconds (second 0 being the start of rplace), and a second int for the milliseconds
        '''
        
        day = int(date.split('-')[2])
        hourlist = hour.split(':')
        hour = int(hourlist[0])
        minute = int(hourlist[1])
        secms = hourlist[2].split('.')
        sec = int(secms[0])

        if(len(secms)==1): ##### special case of ms=0: no "." was found in split
            ms = 0
        else:
            dummy_float_ms = float("0."+secms[1])+1e-5
            ms = int(1000*dummy_float_ms)
            
        second = (day-self.start_day)*86400 + hour*3600 + minute*60 + sec - self.minimal_second
        return (second,ms)

    def condense_part(self, fnum_start=0,fnum_end=79, maxevent=1e12):
        '''
        Transform part of the dataset of pixel changes to a denser file
        
        parameters
        ----------
        start and end file numbers
        
        returns
        -------
        None, juste saves the final data file
        '''
        is2023 = (self.year == 2023)

        if fnum_end>79:
            fnum_end = 79
        ####### output lists. Don't use numpy here, because 'append' is much slower (copies the whole array at each append operation)
        eventNb = []
        sec = []
        user = []
        color = []
        canvasX = []
        canvasY = []
        moderatorEvent = [] ####### boolean saying if this is a rectangle-coverage event from moderator
        colorDict = {} ####### dictionary of existing colors. Only the int keys of these colors are stored
        userDict = {} ####### dictionary of existing users. Only the int keys of these users are stored
        UniqueColNb = 0
        UniqueUserNb = 0    
        event = 0

        print
        ####### loop over files
        for i in range(fnum_start, fnum_end):
            if (event-int(i*1e7)) >= maxevent :
                break

            event = int(i*1e7) ###### so that parallel runs on different files gives different event numbers
            
            if i<=9:
                extra_str = '0'
            else:
                extra_str = ''

            print('open file number ',i)            
            fin = open(self.fullorigfname(self.fname_base + extra_str + str(i) + '.csv'), 'r')

            ####### loop on lines of this file 
            for line in fin.readlines()[1:]: #######skip first line
                if event>=maxevent:
                    break

                if event%300000 == 0:
                    print("Start event #",event)
                    
                l_elem = line.split(maxsplit=2)
                l_elem2 = l_elem[2].split(",")
                ####### color
                col = l_elem2[-1 if is2023 else 2]
                if self.year == 2023:
                    col = col[:-1] # remove "\n" at the end of color
                colidx = colorDict.get(col)
                if colidx==None: ####### case when this color was not added yet in the colorDict
                    colorDict[col] = UniqueColNb
                    colidx = UniqueColNb
                    UniqueColNb += 1
                    
                #######user ID
                userID = (l_elem2[1])[:-2] #######remove "=="
                useridx = userDict.get(userID)
                if useridx==None:  #######case when this user was not added yet in the userDict list
                    userDict[userID] = UniqueUserNb
                    useridx = UniqueUserNb
                    UniqueUserNb += 1

                ####### time (sec,ms)
                (s,millis) = self.simplified_timestamp(l_elem[0], l_elem[1])
                second = float(s+0.001*millis)

                ####### canvas position (X,Y)
                modEventRect = len(l_elem2)>6
                modEventCircle = (len(l_elem2) == 6)
                #if modEventCircle or modEventRect:
                #   print(line)
                #    print(modEventRect, modEventCircle, l_elem2, len(l_elem2))
                if modEventRect:
                    canx1 = int((l_elem2[2 if is2023 else 3])[1:]) # remove "
                    cany1 = int(l_elem2[3 if is2023 else 4])
                    canx2 = int(l_elem2[4 if is2023 else 5])
                    cany2 = int((l_elem2[5 if is2023 else 6])[:-1]) # remove "
                    for x in range(canx1,canx2+1):
                        for y in range(cany1,cany2+1):
                            canvasX.append(x)
                            canvasY.append(y)
                            moderatorEvent.append(True)
                            user.append(useridx)
                            sec.append(second)
                            color.append(colidx)
                            eventNb.append(event)
                elif modEventCircle: # should exist only in 2023
                    xcenter = int((l_elem2[2 if is2023 else 3])[5:]) # remove "{X:
                    ycenter = int((l_elem2[3 if is2023 else 4])[4:]) # remove  Y:
                    radius = int((l_elem2[4 if is2023 else 5])[4:-2]) # remove  R: and }"
                    for x in range(xcenter-radius, xcenter+radius+1):
                        for y in range(ycenter-radius, ycenter+radius+1):
                            if pow((float)(x - xcenter), 2) + pow((float)(y - ycenter), 2) < pow((float)(radius), 2) - 1e-3 :
                                canvasX.append(x)
                                canvasY.append(y)
                                moderatorEvent.append(True)
                                user.append(useridx)
                                sec.append(second)
                                color.append(colidx)
                                eventNb.append(event)
                else: # standard pixel change
                    canx = int((l_elem2[2 if is2023 else 3])[1:]) # remove "
                    cany = int((l_elem2[3 if is2023 else 4])[:-1]) # remove "
                
                    canvasX.append(canx)
                    canvasY.append(cany)
                    moderatorEvent.append(False)
                    user.append(useridx)
                    sec.append(second)
                    color.append(colidx)
                    eventNb.append(event)
                            
                #######ready for next event (=line)
                event += 1
            fin.close()

        print( 'list of existing colors', len(colorDict), colorDict)
        print( 'number of existing users', len(userDict))#, userDict)
        
        ##### Create numpy arrays for output
        sec_out = np.array(sec, dtype='float64')
        eventNb_out = np.array(eventNb, dtype='uint32')
        userIdx_out = np.array(user, dtype='uint32')
        colorIdx_out = np.array(color, dtype='uint8')
        canvasX_out = np.array(canvasX, dtype='int16')
        canvasY_out = np.array(canvasY, dtype='int16')
        modEvent_out = np.array(moderatorEvent, dtype='bool')

        ##### Save arrays to npz file
        str_fileNums = '_files'+str(fnum_start)+'to'+str(fnum_end-1)
        np.savez(self.fullfname('PixelChangesCondensedData'+str_fileNums+'.npz') ,
                seconds = sec_out, eventNumber = eventNb_out, userIndex = userIdx_out, colorIndex = colorIdx_out, pixelXpos = canvasX_out, pixelYpos = canvasY_out, moderatorEvent = modEvent_out )

        ##### Save dictionaries to json file
        fcol = open(self.fullfname("ColorDict"+str_fileNums+".json"),"w")
        fcol.write( json.dumps(colorDict) )
        fcol.close()

        fuser = open(self.fullfname("userDict"+str_fileNums+".json"),"w")
        fuser.write( json.dumps(userDict) )
        fuser.close()
        
        
    def Merging(self): 
        if self.year == 2022:
            fileRangeList = [ ( i*4, (i+1)*4 ) for i in range(0,19) ]
            fileRangeList.append((76,79))
        else:
            fileRangeList = [ ( i*4, (i+1)*4 ) for i in range(0,12) ]
            fileRangeList.append((48, 51))
            fileRangeList.append((51, 53))

        ####### simply concatenate all 1D arrays for the 4 'simple' columns (seconds, pixelXpos, ...)
        print(fileRangeList)
        sec_out = np.concatenate([ a['seconds']
                                for a in ( [np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))
                                            for (fstart,fend) in fileRangeList])
                                ], dtype='float64')
        eventNb_out = np.concatenate([ a['eventNumber']
                                    for a in ( [np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))
                                                for (fstart,fend) in fileRangeList])
                                    ], dtype='uint32')
        canvasX_out = np.concatenate([ a['pixelXpos']
                                    for a in ( [np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))
                                                for (fstart,fend) in fileRangeList])
                                    ], dtype='int16')
        canvasY_out = np.concatenate([ a['pixelYpos']
                                    for a in ( [np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))
                                                for (fstart,fend) in fileRangeList])
                                    ], dtype='int16')
        modEvent_out = np.concatenate([ a['moderatorEvent']
                                    for a in ( [np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))
                                                for (fstart,fend) in fileRangeList])
                                    ], dtype='bool')
        
        ####### Determine total color dictionary (the one from a given few-file output should be enough to have the 32 colors)
        print("Redefining color indices")
        colorDict = {}
        for (fstart,fend) in fileRangeList:
            colorDict = json.load(open(self.fullfname('ColorDict_files'+str(fstart)+'to'+str(fend-1)+'.json')))
            if len(colorDict)>=32:
                shutil.copyfile(self.fullfname('ColorDict_files'+str(fstart)+'to'+str(fend-1)+'.json'), self.fullfname('ColorDict.json'))
                fstart_finDict,_ = (fstart,fend)
                break

        ###### Change color indices (adapt them to the unique color dictionary) in all few-file outputs, then concatenate them
        colorIdx_out = np.array([],dtype='uint8')
        for (fstart,fend) in fileRangeList:
            print('files',fstart,'to',fend-1)
            colarray_tmp = (np.load(self.fullfname('PixelChangesCondensedData_files'+str(fstart)+'to'+str(fend-1)+'.npz'))) ['colorIndex']
            if fstart==fstart_finDict:
                colorIdx_out = np.concatenate((colorIdx_out, colarray_tmp),dtype='uint8')
            else:
                colorDictAlt = json.load(open(self.fullfname('ColorDict_files'+str(fstart)+'to'+str(fend-1)+'.json')))
                DictTranslation = {oldidx:colorDict[color] for color,oldidx in colorDictAlt.items() }  ####### the magic is here: associates each of the 32 old color index with the new color index
                col_tmp2 = [ DictTranslation[val] for val in colarray_tmp ]
                colorIdx_out = np.concatenate((colorIdx_out, np.array(col_tmp2,dtype='uint8')))

        ###### Change userID indices (adapt them to a unique dictionary to be determined) in all few-file outputs, then concatenate them
        print("Redefining user indices")
        userDict = {}
        userIdx_out = np.array([], dtype='uint32')
        for (fstart,fend) in fileRangeList:
            print('files',fstart,'to',fend-1)
            filesNbStr = '_files'+str(fstart)+'to'+str(fend-1)
            userarray_tmp = (np.load(self.fullfname('PixelChangesCondensedData'+filesNbStr+'.npz'))) ['userIndex']
            if fstart==0:
                userDict = json.load(open(self.fullfname('userDict'+filesNbStr+'.json')))
                userIdx_out = userarray_tmp

            else: ###### need to modify userDict indices for files other than the file using the reference dict
                userDictAlt = json.load(open(self.fullfname('userDict'+filesNbStr+'.json')))
                DictTranslation = {}
                for user,oldidx in userDictAlt.items():
                    if user not in userDict: ##### case where this user is NOT already in the reference userDict
                        userDict[user] = len(userDict)
                    DictTranslation[oldidx] = userDict[user] ###### index in the reference userDict, as a function of index in the alternative userDict

                user_tmp2 = [ DictTranslation[val] for val in userarray_tmp ]
                userIdx_out = np.concatenate((userIdx_out, np.array(user_tmp2,dtype='uint32')))


        print('save to',self.fullfname('PixelChangesCondensedData.npz'))
        np.savez(self.fullfname('PixelChangesCondensedData.npz'), seconds = sec_out, eventNumber = eventNb_out, pixelXpos = canvasX_out, pixelYpos = canvasY_out, colorIndex = colorIdx_out, userIndex = userIdx_out, moderatorEvent = modEvent_out)

        ##### Inverse dictionaries for users and colors. Works because indices (dictionary values) are unique
        colorFromIdx = {v: k for k, v in colorDict.items()}
        userIDFromIdx = {int(v): k for k, v in userDict.items()}

        ##### Save dictionaries to json file
        fcol = open(self.fullfname("ColorsFromIdx.json"),"w")
        fcol.write( json.dumps(colorFromIdx) )
        fcol.close()

        fuser = open(self.fullfname("userIDsFromIdx.json"),"w")
        fuser.write( json.dumps(userIDFromIdx) )
        fuser.close()

        fuserIdx = open(self.fullfname("userDict.json"),"w")
        fuserIdx.write( json.dumps(userDict) )
        fuserIdx.close()

    def Sorting(self): #takes 1 to 3min on Guillaume's machine
        '''
        Sort the condensed data according to the 'seconds' array (including the milliseconds as decimals)
        
        parameters
        ----------
        None
        
        returns
        -------
        None
        '''

        arrays = np.load(self.fullfname('PixelChangesCondensedData.npz'))
        sec = arrays['seconds']
        eventNb = arrays['eventNumber']
        canvasX = arrays['pixelXpos']
        canvasY = arrays['pixelYpos']
        col = arrays['colorIndex']
        user = arrays['userIndex']
        modEvent = arrays['moderatorEvent']

        print('Begin sorting the seconds array')
        permut = np.argsort(sec, kind='mergesort') #'mergesort' should be faster on partially sorted arrays
        
        print('Arrange arrays using permutation from sorting \'seconds\' array. Then save to file.')
        sec_out = sec[permut]
        eventNb_out = eventNb[permut]
        canvasX_out = canvasX[permut]
        canvasY_out = canvasY[permut]
        col_out = col[permut]
        user_out = user[permut]
        modEvent_out = modEvent[permut]

        np.savez(self.fullfname('PixelChangesCondensedData_sorted.npz'), seconds = sec_out, eventNumber = eventNb_out, pixelXpos = canvasX_out, pixelYpos = canvasY_out, colorIndex = col_out, userIndex = user_out, moderatorEvent = modEvent_out)

    def remove_duplicates(self): ###### takes ?? min on Guillaume's machine
        '''
        Remove duplicate events from sorted treated data. Using the fact that it's sorted in time.
        
        parameters
        ----------
        None
        
        returns
        -------
        None
        '''
        arrays = np.load(self.fullfname('PixelChangesCondensedData_sorted.npz'))
        sec = arrays['seconds']
        eventNb = arrays['eventNumber']
        canvasX = arrays['pixelXpos']
        canvasY = arrays['pixelYpos']
        col = arrays['colorIndex']
        user = arrays['userIndex']
        modEvent = arrays['moderatorEvent']

        to_delete = [] ###### indexes of duplicates, to be deleted later in all arrays
        for i in range(0,sec.size-1):
            if i%(int(1e7)) == 0:                                                                                                                                                                                                                            print('test event',i)

            if modEvent[i]: #no duplicates expected for moderated events
                continue
            
            iend = i+1
            while (sec[iend] - sec[i]) < 1e-4 and iend<sec.size : ###### check if seconds (and ms) is the same at indices i and iend
                #print('same time and canvasY for events',i,iend, ' time =', textTimestamp_fromSeconds(sec[i]),'x,y = ',canvasX[i],canvasY[i],'   x,y for iend = ',canvasX[iend],canvasY[iend])
                if (canvasY[iend] == canvasY[i]) and (canvasX[iend] == canvasX[i]) and (user[iend] == user[i]) and (col[iend] == col[i]) and (modEvent[iend] == modEvent[i]):
                    #print('need to delete element',iend,'already present at position',i)
                    to_delete.append((iend,i))
                iend += 1

        sec = np.delete(sec, to_delete)
        eventNb = np.delete(eventNb, to_delete)
        canvasX = np.delete(canvasX, to_delete)
        canvasY = np.delete(canvasY, to_delete)
        col = np.delete(col, to_delete)
        user = np.delete(user, to_delete)
        modEvent = np.delete(modEvent, to_delete)

        print('number of duplicates deleted =',len(to_delete))
                
        np.savez(self.fullfname('PixelChangesCondensedData_sorted.npz'), seconds = sec, eventNumber = eventNb, pixelXpos = canvasX, pixelYpos = canvasY, colorIndex = col, userIndex = user, moderatorEvent = modEvent)

    def tag_hidden_mod_changes(self): 
        '''
        Switches 'moderatorEvent' to 1 when the user hashtag is that of a user that has done another obvious mod change
        
        parameters
        ----------
        None
        
        returns
        -------
        None
        '''
        arrays = np.load(self.fullfname('PixelChangesCondensedData_sorted_old.npz'))
        sec = arrays['seconds']
        eventNb = arrays['eventNumber']
        canvasX = arrays['pixelXpos']
        canvasY = arrays['pixelYpos']
        col = arrays['colorIndex']
        user = arrays['userIndex']
        modEvent = arrays['moderatorEvent']

        usermod = np.unique(user[modEvent == 1])

        modEvent[np.where(np.isin(user, usermod))] = True

        np.savez(self.fullfname('PixelChangesCondensedData_sorted_AllModChanges.npz'), seconds = sec, eventNumber = eventNb, pixelXpos = canvasX, pixelYpos = canvasY, colorIndex = col, userIndex = user, moderatorEvent = modEvent)
        
    def misc_checks(self):
        ###### function just to check the output data
        arrays = np.load(self.fullfname('PixelChangesCondensedData_sorted.npz'))
        sec = arrays['seconds']
        eventNb = arrays['eventNumber']
        canvasX = arrays['pixelXpos']
        canvasY = arrays['pixelYpos']
        col = arrays['colorIndex']
        user = arrays['userIndex']
        modEvent = arrays['moderatorEvent']

        userDict = json.load(open(self.fullfname('userIDsFromIdx.json')))
        colorDict = json.load(open(self.fullfname('ColorsFromIdx.json')))

        ###### Print out range of events from final file
        for i in range(int(5e7)+0,int(5e7)+100):
            print(i, self.textTimestamp_fromSeconds(sec[i]),'{:.3f}'.format(sec[i]),eventNb[i], canvasX[i],canvasY[i],  colorDict[str(col[i])], userDict[str(user[i])], modEvent[i])

        
        ###### print events from a given user
        for i in range(0,sec.size):
            if i%(int(1e7)) == 0:
                print('test event',i)
            if userDict[str(user[i])] == "pKhpk2fRigobnKBW/bvGGNifViNGiZCSBmZFFSosTCDUSJb+0FDWMf081vyrSyBBxQ/Xkb573neTu/0Y25qK8A":#"gs1en6zLLigg2LnGvHyWLQBJwXzhe3ENPRmZQ/3EU9oVsxzXMAivwZJ/3E5yoLG9HC3upsRtqw9ftb4r0ytRcQ" :
                print(i)
                print(self.textTimestamp_fromSeconds(sec[i]),'{:.3f}'.format(sec[i]),eventNb[i], canvasX[i],canvasY[i],  colorDict[str(col[i])], userDict[str(user[i])])
        
        ###### Count number of moderator events
        modEventNumbers = set()
        for i in range(0,sec.size-1):
            if modEvent[i]:
                modEventNumbers.add(eventNb[i])
        print('number of moderator events =',len(modEventNumbers))

        '''
        ###### check if there are no duplicates. Using the fact that it's sorted in time
        for i in range(0,sec.size-1):
            if modEvent[i]: #no duplicates expected for moderated events
                continue
            iend = i+1
            while (sec[iend] - sec[i]) < 1e-4 and iend<sec.size : ###### check if seconds (and ms) is the same at indices i and iend
                if (canvasY[iend] == canvasY[i]) and (canvasX[iend] == canvasX[i]) and (user[iend] == user[i]) and (col[iend] == col[i]) and (modEvent[iend] == modEvent[i]):
                    print ('found a duplicate !! Positions in array : ',i,iend)
                    print(i, self.textTimestamp_fromSeconds(sec[i]),'{:.3f}'.format(sec[i]),eventNb[i], canvasX[i],canvasY[i],  colorDict[str(col[i])], userDict[str(user[i])], modEvent[i])
                    print(iend, self.textTimestamp_fromSeconds(sec[iend]),'{:.3f}'.format(sec[iend]),eventNb[iend], canvasX[iend],canvasY[iend],  colorDict[str(col[iend])], userDict[str(user[iend])], modEvent[i])
                iend += 1
        '''
        
    def clean_data_dir(self):

        for f in glob.glob(self.fullfname('PixelChangesCondensedData_files*.npz')):
            os.remove(f)
        os.remove(self.fullfname('PixelChangesCondensedData.npz'))
        for f in glob.glob(self.fullfname('userDict_files*.json')):
            os.remove(f)
        for f in glob.glob(self.fullfname('ColorDict_files*.json')):
            os.remove(f)
